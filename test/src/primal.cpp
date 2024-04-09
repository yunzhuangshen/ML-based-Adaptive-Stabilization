/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*                                                                           */
/*                  This file is part of the program and library             */
/*         SCIP --- Solving Constraint Integer Programs                      */
/*                                                                           */
/*    Copyright (C) 2002-2019 Konrad-Zuse-Zentrum                            */
/*                            fuer Informationstechnik Berlin                */
/*                                                                           */
/*  SCIP is distributed under the terms of the ZIB Academic License.         */
/*                                                                           */
/*  You should have received a copy of the ZIB Academic License              */
/*  along with SCIP; see the file COPYING. If not visit scip.zib.de.         */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

/**@file   heur_init.c
 * @brief  initial primal heuristic for the vertex coloring problem
 * @author Gerald Gamrath
 *
 * This file implements a heuristic which computes a starting solution for the coloring problem. It
 * therefore computes maximal stable sets and creates one variable for each set, which is added to the
 * LP.
 *
 * The heuristic is called only one time: before solving the root node.
 *
 * It checks, whether a solution-file was read in and a starting solution already exists.  If this
 * is not the case, an initial possible coloring is computed by a greedy method.  After that, a
 * tabu-search is called, which tries to reduce the number of colors needed. The tabu-search algorithm
 * follows the description in
 *
 * "A Survey of Local Search Methods for Graph Coloring"@n
 * by P. Galinier and A. Hertz@n
 * Computers & Operations Research, 33 (2006)
 *
 * The tabu-search works as follows: given the graph and a number of colors it tries to color the
 * nodes of the graph with at most the given number of colors.  It starts with a random coloring. In
 * each iteration, it counts the number of violated edges, that is, edges for which both incident
 * nodes have the same color. It now switches one node to another color in each iteration, taking
 * the node and color, that cause the greatest reduction of the number of violated edges, or if no
 * such combination exists, the node and color that cause the smallest increase of that number.  The
 * former color of the node is forbidden for a couple of iterations in order to give the possibility
 * to leave a local minimum.
 *
 * As long as the tabu-search finds a solution with the given number of colors, this number is reduced
 * by 1 and the tabu-search is called another time. If no coloring was found after a given number
 * of iterations, the tabu-search is stopped and variables for all sets of the last feasible coloring
 * are created and added to the LP (after possible extension to maximal stable sets).
 *
 * The variables of these sets result in a feasible starting solution of the coloring problem.
 *
 * The tabu-search can be deactivated by setting the parameter <heuristics/initcol/usetabu> to
 * false.  The number of iterations after which the tabu-search stops if no solution was yet found
 * can be changed by the param <heuristics/initcol/maxiter>. A great effect is also obtained by
 * changing the parameters <heuristics/initcol/tabubase> and <heuristics/initcol/tabugamma>, which
 * determine the number of iterations for which the former color of a node is forbidden; more
 * precisely, this number is \<tabubase\> + ncritical * \<tabugamma\>, where ncritical is the number
 * of nodes, which are incident to violated edges.  Finally, the level of output and the frequency of
 * status lines can be changed by <heuristics/initcol/output> and <heuristics/initcol/dispfreq>.
 */

/*---+----1----+----2----+----3----+----4----+----5----+----6----+----7----+----8----+----9----+----0----+----1----+----2*/

#include <assert.h>
#include "primal.h"

#define DEFAULT_MAXITER    100000
#define DEFAULT_TABUBASE   50
#define DEFAULT_TABUGAMMA  0.9
using namespace std;

/** computes the number of violated edges, that means the number of edges (i,j) where i and j have the same color */
int getNViolatedEdges(
   Instance*             graph,              /**< the graph */
   int*                  colors              /**< colors of the nodes */
   )
{
   int nnodes;
   int i;
   int* j;
   int cnt;

   assert(graph != NULL);
   assert(colors != NULL);

   /* get the number of nodes */
   nnodes = graph->n_nodes;
   cnt = 0;

   /* count the number of violated edges, only consider edges (i,j) with i > j since the graph is undirected bu */
   for( i = 0; i < nnodes; i++ )
   {
      if (graph->adj_list[i].empty()) continue;
      for( j = graph->getFirstAdjedge(i);  j <= graph->getLastAdjedge(i) && *j < i; j++ )
      {
         if( colors[i] == colors[*j] )
            cnt++;
      }
   }
   return cnt;
}


/*
 * Local methods
 */

/** checks whether one of the nodes has no color respectively has color -1 in the given array */
bool hasUncoloredNode(
   int                   nnodes,             /**< the graph that should be colored */
   int*                  colors              /**< array of ints representing the colors */
   )
{
   int i;

   assert(colors != NULL);

   for( i = 0; i < nnodes; i++)
   {
      /* node not yet colored */
      if(colors[i] == -1)
      {
         return true;
      }
   }
   return false;
}


/** computes a stable set with a greedy-method and colors its nodes */
void greedyStableSet(
   Instance*             graph,              /**< pointer to graph data structure */
   int*                  colors,             /**< array of ints representing the different colors, -1 means uncolored */
   int                   nextcolor           /**< color in which the stable set will be colored */
   )
{
   bool indNode;
   int nnodes;
   int i;
   int j;
   int* degrees;
   int nstablesetnodes;

   assert(graph != NULL);
   assert(colors != NULL);

   /* get number of nodes */
   nnodes = graph->n_nodes;

   /* get the  degrees and weights for the nodes in the graph */
   degrees = graph->degree.data();

   int stablesetnodes[nnodes];
   int values[nnodes];
   int sortednodes[nnodes];

   /* set values to the nodes which are used for sorting them */
   /* value = degree of the node + number of nodes if the node is uncolored,
      therefore the colored nodes have lower values than the uncolored nodes */
   for( i = 0; i < nnodes; i++ )
   {
      sortednodes[i] = i;
      values[i] = degrees[i] + ( colors[i] == -1 ? nnodes : 0);
      // cout << colors[i] << " " << values[i] << "\n";
   }


   /* sort the nodes in non-increasing order*/
   sort_indexes_noninc(values, sortednodes, nnodes);

   // for (i = 0; i < nnodes; i++ )
   //    cout << values[i] << " " << sortednodes[i] << " " << colors[i] << "\n";
   // cout << "\n";

   /* insert first node */
   stablesetnodes[0] = sortednodes[0];
   nstablesetnodes = 1;
   for( i = 1; i < nnodes; i++)
   {
      if( colors[sortednodes[i]] != -1 )
      {
         break;
      }
      indNode = true;
      for( j = 0; j < nstablesetnodes; j++ )
      {
         if( graph->is_edge(sortednodes[i], stablesetnodes[j]) )
         {
            indNode = false;
            break;
         }
      }
      if( indNode == true )
      {
         stablesetnodes[nstablesetnodes] = sortednodes[i];
         nstablesetnodes++;
      }

   }
   for( i = 0; i < nstablesetnodes; i++ )
   {
      if (colors[stablesetnodes[i]]!=-1){
         cout << colors[stablesetnodes[i]]<< "\n";
      }
      assert(colors[stablesetnodes[i]] == -1);
      colors[stablesetnodes[i]] = nextcolor;
   }
}


/** computes the initial coloring with a greedy method */
void greedyInitialColoring(
   Instance*             graph,              /**< pointer to graph data structure */
   int*                  colors,             /**< array of ints representing the different colors */
   int*                  ncolors             /**< number of colors needed */
   )
{
   int nnodes;
   int i;
   int color;
   assert(colors != NULL);

   nnodes = graph->n_nodes;
   assert(nnodes > 0);

   for( i = 0; i < nnodes; i++ )
   {
      colors[i] = -1;
   }

   color = 0;
   /* create stable sets until all Nodes are covered */
   while( hasUncoloredNode(nnodes, colors) )
   {
      greedyStableSet(graph, colors, color);
      color++;
   }
   *ncolors = color;
}


/** runs tabu coloring heuristic, gets a graph and a number of colors
 *  and tries to color the graph with at most that many colors;
 *  starts with a random coloring and switches one node to another color in each iteration,
 *  forbidding the old color for a couple of iterations
 */
void runTabuCol(
   Instance*             graph,              /**< the graph, that should be colored */
   int                   seed,               /**< seed for the first random coloring */
   int                   maxcolors,          /**< number of colors, which are allowed */
   int*                  colors,             /**< output: the computed coloring */
   bool*                 success             /**< pointer to store if something went wrong */
   )
{
   int nnodes;
   int obj;
   int bestobj;
   int i;
   int j;
   int node1;
   int node2;
   int color1;
   int color2;
   int* firstedge;
   int* lastedge;
   bool restrictive;
   int iter;
   int minnode;
   int mincolor;
   int minvalue;
   int ncritical;
   bool aspiration;
   int d;
   int oldcolor;

   assert(graph != NULL);
   assert(success != NULL);

   printf("Running tabu coloring with maxcolors = %d...\n", maxcolors);

   /* get size */
   nnodes = graph->n_nodes;

   srand( seed ); /*lint !e732*/

   /* init random coloring */
   for( i = 0; i < nnodes; i++ )
   {
      int rnd = rand();
      colors[i] = rnd % maxcolors;
      assert( 0 <= colors[i] && colors[i] < maxcolors );
   }

   int tabu[nnodes][maxcolors];
   int adj[nnodes][maxcolors];

   for( i = 0; i < nnodes; i++ )
   {
      for( j = 0; j < maxcolors; j++ )
      {
         tabu[i][j] = 0;
         adj[i][j] = 0;
      }
   }

   /* objective */
   obj = 0;

   /* init adj-matrix and objective */
   for( node1 = 0; node1 < nnodes; node1++ )
   {
      if (graph->adj_list[node1].empty()) continue;
      color1 = colors[node1];
      firstedge = graph->getFirstAdjedge(node1);
      lastedge = graph->getLastAdjedge(node1);
      while(  firstedge <= lastedge )
      {
         node2 = *firstedge;
         color2 = colors[node2];
         assert( 0 <= color2 && color2 < maxcolors );
         (adj[node1][color2])++;
         if( color1 == color2 )
            obj++;
         firstedge++;
      }
   }
   assert( obj % 2 == 0 );
   obj = obj / 2;
   assert( obj == getNViolatedEdges(graph, colors) );

   bestobj = obj;
   restrictive = false;
   iter = 0;
   if( obj > 0 )
   {
      /* perform predefined number of iterations */
      for( iter = 1; iter <= DEFAULT_MAXITER; iter++ )
      {
         /* find best 1-move among those with critical vertex */
         minnode = -1;
         mincolor = -1;
         minvalue = nnodes * nnodes;
         ncritical = 0;
         for( node1 = 0; node1 < nnodes; node1++ )
         {
            aspiration = false;
            color1 = colors[node1];
            assert( 0 <= color1 && color1 < maxcolors );

            /* if node is critical (has incident violated edges) */
            if( adj[node1][color1] > 0 )
            {
               ncritical++;
               /* check all colors */
               for( j = 0; j < maxcolors; j++ )
               {
                  /* if color is new */
                  if( j != color1 )
                  {
                     /* change in the number of violated edges: */
                     d = adj[node1][j] - adj[node1][color1];

                     /* 'aspiration criterion': stop if we get feasible solution */
                     if( obj + d == 0 )
                     {
                        printf("   Feasible solution found after %d iterations!\n\n", iter);
                        minnode = node1;
                        mincolor = j;
                        minvalue = d;
                        aspiration = true;
                        break;
                     }

                     /* if not tabu and better value */
                     if( tabu[node1][j] < iter &&  d < minvalue )
                     {
                        minnode = node1;
                        mincolor = j;
                        minvalue = d;
                     }
                  }
               }
            }
            if( aspiration )
            break;
         }

         /* if no candidate could be found - tabu list is too restrictive: just skip current iteration */
         if( minnode == -1 )
         {
            restrictive = true;
            continue;
         }
         assert( minnode != -1 );
         assert( mincolor >= 0 );

         /* perform changes */
         assert( colors[minnode] != mincolor );
         oldcolor = colors[minnode];
         colors[minnode] = mincolor;
         obj += minvalue;
         // assert( obj == getNViolatedEdges(graph, colors) );
         if( obj < bestobj )
            bestobj = obj;

         // if( heurdata->output == 2 && (iter) % (heurdata->dispfreq) == 0 )
         // {
         //    printf("Iter: %d  obj: %d  critical: %d   node: %d  color: %d  delta: %d\n", iter, obj, ncritical, minnode,
         //          mincolor, minvalue);
         // }

         /* terminate if valid coloring has been found */
         if( obj == 0 )
            break;

         /* update tabu list */
         assert( tabu[minnode][oldcolor] < iter );
         tabu[minnode][oldcolor] = iter + (DEFAULT_TABUBASE) + (int) (((double) ncritical) * (DEFAULT_TABUGAMMA));

         if (graph->adj_list[minnode].empty()) continue;
         /* update adj matrix */
         for( firstedge = graph->getFirstAdjedge(minnode); firstedge <= graph->getLastAdjedge(minnode); firstedge++ )
         {
            (adj[*firstedge][mincolor])++;
            (adj[*firstedge][oldcolor])--;
         }
      }
   }

   if(bestobj != 0 )
   {
      printf("   No feasible solution found after %d iterations!\n\n", iter-1);
   }

   /* check whether valid coloring has been found */
   *success = (obj == 0);
}


void runHeur(
   Instance*             graph,
   int                   seed,
   vector<vector<int>>& cols)
{
   int i;
   int j;
   int k;
   int nnodes;
   bool stored;
   bool success;
   bool indnode;
   int ncolors;
   int nstablesetnodes;

   nnodes = graph->n_nodes;

   /* create stable sets if no solution was read */
   int colors[nnodes];
   int bestcolors[nnodes];

   /* compute an initial coloring with a greedy method */
   greedyInitialColoring(graph, bestcolors, &ncolors);

   /* try to find better colorings with tabu search method */
   success = true;
   while( success )
   {
      ncolors--;
      runTabuCol(graph, seed, ncolors, colors, &success);

      if( success )
      {
         for( i = 0; i < nnodes; i++ )
         {
            bestcolors[i] = colors[i];
         }
      }else
         ncolors++;

   }
   
   /* create vars for the computed coloring */
   for( i = 0; i < ncolors; i++ )
   {
      /* save nodes with color i in the array colors and the number of such nodes in nstablesetnodes */
      nstablesetnodes = 0;
      for( j = 0; j < nnodes; j++ )
      {
         if( bestcolors[j] == i )
         {
            colors[nstablesetnodes] = j;
            nstablesetnodes++;
         }
      }
      /* try to add more nodes to the stable set without violating the stability */
      for( j = 0; j < nnodes; j++ )
      {
         indnode = true;
         for( k = 0; k < nstablesetnodes; k++ )
         {
            if( j == colors[k] || graph->is_edge(j, colors[k]) )
            {
               indnode = false;
               break;
            }
         }
         if( indnode == true )
         {
            colors[nstablesetnodes] = j;
            nstablesetnodes++;
         }
      }

      vector<int> col;
      for (j = 0; j < nstablesetnodes; j++){
         col.push_back(colors[j]);
      }
      cols.push_back(col);
   }
}
