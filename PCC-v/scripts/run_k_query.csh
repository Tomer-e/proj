#! /bin/csh -f

set j = 1
while ( $j <= 11 )
  python3 correct_queries/marabou_K_query3.py model/output_graph.pb $j
  @ j++
end
