#include "crush_map.h"

struct node{
  int id;
  int pg_num;
  node(int id, int num) : id(id), pg_num(num) {}
  bool operator < (const node& a) const {
    return pg_num < a.pg_num; //大顶堆
  }
};

void crush_map::gen_map(int osd_num, int osd_per_host) {
  bucket b(bucket_id--);
  b.weight = 0;
  b.pg_num = 0;
  b.primary_pg = 0;
  root_weight = 0;
  for (int i = 0; i < osd_num; i++) {
    osd_weight.push_back(1);
    osd_pg_num.push_back(0);
    osd_primary_pg_num.push_back(0);
    osd_bucket.push_back(b.id);
    if (i%osd_per_host == 0 && i > 0) {
      root_weight += b.weight;
      buckets.push_back(b);
      b.id = bucket_id--;
      b.weight = 0;
      b.item.clear();
    }
    b.item.push_back(osd_id++);
    b.weight++;
  }
  root_weight += b.weight;
  buckets.push_back(b);
}

void crush_map::set_crushmap_pg_target(unsigned pool_size, unsigned pool_pg_num) {
  int pool_pgs = (pool_size-1) * pool_pg_num;
  int pool_primary_num = pool_pg_num;
  float pgs_per_weight = 1.0 * pool_pgs / root_weight;
  float primary_per_weight = 1.0 * pool_primary_num / root_weight;
  int duplicate_num, primary_num;
  for(auto & b : buckets) {
    b.pg_num = ceil(b.weight * pgs_per_weight);
    b.primary_pg = ceil(b.weight * primary_per_weight);

    for(int i = 0; i < b.item.size(); i++) {
      int item = b.item[i];
      duplicate_num = ceil(osd_weight[item] * pgs_per_weight);
      primary_num = ceil(osd_weight[item] * primary_per_weight);
      osd_pg_num[item] = duplicate_num;   
      osd_primary_pg_num[item] = primary_num;
    }
  }
}
void crush_map::adjust_crushmap_pg_target(unsigned pool_size, unsigned pool_pg_num,
                                          map<int, set<int>>& primary_change,                      
                                          map<int, set<int>>& duplicate_change) {
  int pool_pgs = (pool_size-1) * pool_pg_num;
  int pool_primary_num = pool_pg_num;
  float pgs_per_weight = 1.0 * pool_pgs / root_weight;
  float primary_per_weight = 1.0 * pool_primary_num / root_weight;
  int duplicate_num, primary_num;
  for(auto & b : buckets) {
    duplicate_num = ceil(b.weight * pgs_per_weight);
    primary_num = ceil(b.weight * primary_per_weight);
    b.pg_num = duplicate_num;
    b.primary_pg = primary_num;

    for(int i = 0; i < b.item.size(); i++) {
      int item = b.item[i];
      duplicate_num = ceil(osd_weight[item] * pgs_per_weight);
      primary_num = ceil(osd_weight[item] * primary_per_weight);
      if(osd_pg_num[item] != duplicate_num) {
        duplicate_change[duplicate_num - osd_pg_num[item]].insert(item);
      }
      if(osd_primary_pg_num[item] != primary_num) {
        primary_change[primary_num - osd_primary_pg_num[item]].insert(item);
      }
      osd_pg_num[item] = duplicate_num;   
      osd_primary_pg_num[item] = primary_num;
    }
  }
}

void crush_map::init_pg_mappings(unsigned pool_size, unsigned pool_pg_num) {
  priority_queue<node> q;
  priority_queue<node> primary_q;
  for(auto& b : buckets) {
    q.push(node(b.id, b.pg_num));
    primary_q.push(node(b.id, b.primary_pg));
  }
  for(unsigned pg = 0; pg < pool_pg_num; pg++) { //choose bucket
    vector<int> out_bucket;
    vector<node> t;
    for(int i = 0; i < pool_size && !q.empty();) {
      if(i == 0) { //primary
        t.push_back(primary_q.top()); primary_q.pop();
        out_bucket.push_back(t[i].id);
        buckets_to_primary_pgs[t[i].id].insert(pg);
        t[i].pg_num--;
      } else {
        t.push_back(q.top()); q.pop();
        if(find(out_bucket.begin(), out_bucket.end(), t.back().id) == out_bucket.end()) {
          out_bucket.push_back(t.back().id);
          buckets_to_pgs[t.back().id].insert(pg);
          t.back().pg_num--;
        } else {
          continue;
        }
      }
      i++;
    }
    for(int i = 0; i < t.size(); i++) {
      if(t[i].pg_num > 0) {
        if(i == 0) {
          primary_q.push(t[i]);
        } else {
          q.push(t[i]);
        }
      }
    }
  }
  
  for(auto& b : buckets) {
    while(!q.empty()) q.pop();
    while(!primary_q.empty()) primary_q.pop();
    for(int i = 0; i < b.item.size(); i++) {
      q.push(node(b.item[i], osd_pg_num[b.item[i]]));
      primary_q.push(node(b.item[i], osd_primary_pg_num[b.item[i]]));
    }
    for(auto pg : buckets_to_primary_pgs[b.id]) {
      node n = primary_q.top(); primary_q.pop();      
      if(pgs_to_osd[pg].empty()) {
        pgs_to_osd[pg].push_back(n.id);
      } else {
        pgs_to_osd[pg][0] = n.id;
      }
      osd_to_primary_pgs[n.id].insert(pg);
      n.pg_num--;                      
      if(n.pg_num > 0) {               
        primary_q.push(n);                     
      }
    }

    for(auto pg : buckets_to_pgs[b.id]) {
      node n = q.top(); q.pop();
      if(pgs_to_osd[pg].empty()) {
        pgs_to_osd[pg].push_back(-1);
      }
      pgs_to_osd[pg].push_back(n.id);
      osd_to_pgs[n.id].insert(pg);
      n.pg_num--;
      if(n.pg_num > 0) {
        q.push(n);
      }
    }
  }
}

void crush_map::dump_map() {
  for (auto b : buckets) {
    cout << b.id << "(" << b.weight << ", " << b.pg_num << ", "<< b.primary_pg << ")"<< endl;
    cout << "\t";
    for(auto& i : b.item) {
      cout << i << "(" << osd_weight[i] << ", " << osd_pg_num[i] << ", " << osd_primary_pg_num[i] << ")" << " ";
    }
    cout << endl;
  }
}

void crush_map::dump_result(int pg_nums) {
  for(int i = 0; i < pg_nums; i++) {
    for(int j = 0; j < pgs_to_osd[i].size(); j++) {
      cout << pgs_to_osd[i][j] << " ";
    }
    cout << endl;
  }
}

void crush_map::apply_map_change(map_change& change) {
  map<int, set<int>> duplicate_change; //change_num->item
  map<int, set<int>> primary_change; //change_num->item;

  //reweight
  for(auto p : change.osd_weight_change) {
    int osd_bucket = p.first;
    int osd = p.second.first;
    int weight = p.second.second;

    buckets[-1-osd_bucket].weight += (weight - osd_weight[osd]);
    root_weight += (weight-osd_weight[osd]);
    osd_weight[osd] = weight;
  }
  
  adjust_crushmap_pg_target(change.pool_size, change.pool_pg_num,
                            primary_change, duplicate_change);
  
  //for(auto item : change_pg_num) {
  //  cout << item.first << " " << item.second << endl;
  //}
  for(auto item : primary_change) {
    cout << item.first << ":" << endl;
    //for(auto i : item.second) {
    //  cout << "\t" << i << endl;
    //}
  }

  //adjust pg mappings
  //primary
  while(true) {
    auto over_it = primary_change.begin();
    auto under_it = primary_change.rbegin();
    cout << over_it->first << " " << under_it->first << endl;
    if(over_it->first >= 0 && under_it->first <= 1) {
      break; //aready balance
    }
    
    if(over_it->first < 0) { //overfull
      int over_osd = *over_it->second.begin(); over_it->second.erase(over_it->second.begin());
      int under_osd = *under_it->second.begin(); under_it->second.erase(under_it->second.begin());
      int over_num = over_it->first;
      int under_num = under_it->first;

      while(over_num && under_num) {

        
      }
    }
  }
  
  //duplicate
  
}
