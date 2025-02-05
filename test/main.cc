#include "crush_map.h"

int main()
{
  int pg_nums = 30000;
  int pool_size = 3;
  crush_map map;
  map.gen_map(1000, 10);
  map.set_crushmap_pg_target(pool_size, pg_nums);
  //map.init_pg_mappings(pool_size, pg_nums);
  //map.dump_result(pg_nums);
  //map.dump_map();
  map_change c;
  c.osd_weight_change.push_back({-1, {0, 0}});
  c.osd_weight_change.push_back({-1, {1, 0}});
  c.pool_size = pool_size;
  c.pool_pg_num = pg_nums;
  map.apply_map_change(c);
  //map.dump_map();

  return 0;
}

