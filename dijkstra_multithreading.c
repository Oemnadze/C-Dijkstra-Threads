#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
const int INFINITY = 2000000000;
const int MAX_VERTEXES = 1000;

int n_vertices;
int source;

void init(int graph[MAX_VERTEXES][MAX_VERTEXES]) {
	printf("Enter Number of vertices, then adjacent matrix representation of the graph and at last an index of the source node!\n");
	scanf("%d", &n_vertices);
	printf("\n");

	for (int i = 0; i < n_vertices; i++) {
		for (int j = 0; j < n_vertices; j++) {
			scanf("%d", &graph[i][j]);
		}
	}

	scanf("%d", &source);
}

int* find_distances(int source, int graph[MAX_VERTEXES][MAX_VERTEXES]) {
	int* used = malloc(n_vertices * sizeof(int));
	int* distances = malloc(n_vertices * sizeof(int));

	for (int i = 0; i < n_vertices; i++) {
		used[i] = 0;
	}
	used[source] = 1;

	for (int i = 0; i < n_vertices; i++) {
		distances[i] = graph[source][i];
	}

	int left, right, id, min_dist, min_dist_vertex, num_threads;

	# pragma omp parallel private (left, right, id) \
	shared (used, min_dist, min_dist_vertex, distances, num_threads) 
	{
		id = omp_get_thread_num();
		num_threads = omp_get_num_threads();
		left = (id * n_vertices) / num_threads;
		right = ((id + 1) * n_vertices) / num_threads - 1;

		for (int i = 1; i < n_vertices; i++) {
			# pragma omp single // just one thread gets into here
			{
				min_dist = INFINITY;
				min_dist_vertex = -1;
			}

			// find closest unvisited node
			int temp_min_dist = INFINITY, temp_min_dist_vertex = -1;
			for (int j = left; j <= right; j++) {
				if (!used[j] && (distances[j] < temp_min_dist)) {
					temp_min_dist = distances[j];
					temp_min_dist_vertex = j;
				}
			}

			# pragma omp critical // one thread at a time
			{ 
				if (temp_min_dist < min_dist) {
					min_dist = temp_min_dist;
					min_dist_vertex = temp_min_dist_vertex;
				}
			}

			# pragma omp barrier // waits for all the threads to process till here

			# pragma omp single 
			{
				if (min_dist_vertex != -1) {
					used[min_dist_vertex] = 1;
				}
			}

			# pragma omp barrier

			if (min_dist_vertex != -1) {
				for (int j = left; j <= right; j++) {
					if (!used[j] && (distances[min_dist_vertex] + graph[min_dist_vertex][j] < distances[j])) {
						distances[j] = distances[min_dist_vertex] + graph[min_dist_vertex][j];
					}
				}
			}

			# pragma omp barrier
		}
	}

	free(used);

	return distances;
}

int main() {

	int graph[MAX_VERTEXES][MAX_VERTEXES]; // 2-dimensional array. Element in i-th row and j-th column is identified as graph[i * n_vertices + j];

	init(graph);

	int* distances = find_distances(source, graph);

	printf("Printing distances from %d-th vertex to every other vertex.\n-1 means that there is no way to get to i-th vertex from %d-th vertex\n", source, source);
	for (int i = 0; i < n_vertices; i++) {
		printf("Distance from %d to %d is %d\n", source, i, (distances[i] != INFINITY) ? distances[i] : -1);
	}

	free(distances);

	return 0;
}