#pragma once
#include <algorithm>
#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <cmath>
#include <queue>
using namespace std;

struct bucket {
  int id;
  int weight;
  int pg_num;
  int primary_pg;
  vector<int> item;
  bucket(int id) : id(id), weight(0), pg_num(0), primary_pg(0) {}
  bucket() : id(0), weight(0), pg_num(0), primary_pg(0) {}
};

struct map_change {
  int pool_size;
  int pool_pg_num;
  vector<pair<int, int>> osd_remove;
  vector<pair<int, int>> osd_add; 
  vector<pair<int, pair<int, int>>> osd_weight_change;
};

class crush_map
{
public:
  crush_map() {
    osd_id = 0;
    bucket_id = -1;
  }
  ~crush_map() {}

private:
  int osd_id;
  int bucket_id;
  vector<int> osd_weight;
  vector<int> osd_pg_num;
  vector<int> osd_primary_pg_num;
  vector<int> osd_bucket; //the bucket og osd

  vector<bucket> buckets;
  int root_weight;
  int max_osd;
  int max_bucket;


  map<int, vector<int>> pgs_to_osd;

  map<int, set<int>> buckets_to_pgs;
  map<int, set<int>> buckets_to_primary_pgs;
  map<int, set<int>> osd_to_pgs;
  map<int, set<int>> osd_to_primary_pgs;

public:
  void gen_map(int osd_num, int osd_per_host);
  void set_crushmap_pg_target(unsigned pool_size, unsigned pool_pg_num);
  void adjust_crushmap_pg_target(unsigned pool_size, unsigned pool_pg_num,
                                 map<int, set<int>>& primary_change,
                                 map<int, set<int>>& duplicate_change);
  void init_pg_mappings(unsigned pool_size, unsigned pool_pg_num);
  void dump_map();
  void dump_result(int pg_nums);
  void apply_map_change(map_change& change);
};

