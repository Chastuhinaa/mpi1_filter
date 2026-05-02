
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <windows.h>

static const int DEFAULT_N   = 20;
static const int VALUE_RANGE = 200; 
static const int ROOT        = 0;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    SetConsoleOutputCP(65001);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = (argc > 1) ? atoi(argv[1]) : DEFAULT_N;
    if (n < size) n = size;   

    int base  = n / size;
    int extra = n % size;  

    std::vector<int> sendcounts(size), displs(size);
    for (int i = 0; i < size; ++i) {
        sendcounts[i] = base + (i < extra ? 1 : 0);
        displs[i]     = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
    }

    int local_n = sendcounts[rank];
    std::vector<int> local_data(local_n);

    std::vector<int> global_data;
    double mean = 0.0;

    if (rank == ROOT) {
        global_data.resize(n);
        srand(static_cast<unsigned>(time(nullptr)));
        for (int i = 0; i < n; ++i)
            global_data[i] = (rand() % VALUE_RANGE) - VALUE_RANGE / 2;

        double sum = std::accumulate(global_data.begin(), global_data.end(), 0LL);
        mean = sum / n;

        printf("=== Паралельна фільтрація масиву (MPI) ===\n");
        printf("Кількість процесів : %d\n", size);
        printf("Розмір масиву      : %d\n", n);
        printf("\nПочатковий масив:\n  ");
        for (int i = 0; i < n; ++i) printf("%4d", global_data[i]);
        printf("\n");
        printf("Середнє арифметичне: %.2f\n", mean);
        printf("Умова фільтрації   : елемент > %.2f\n\n", mean);
    }

    MPI_Bcast(&mean, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    MPI_Scatterv(
        global_data.data(), sendcounts.data(), displs.data(), MPI_INT,
        local_data.data(), local_n, MPI_INT,
        ROOT, MPI_COMM_WORLD
    );

    double t_start = MPI_Wtime();

    std::vector<int> local_result;
    local_result.reserve(local_n);
    for (int v : local_data)
        if (v > mean) local_result.push_back(v);

    double t_end = MPI_Wtime();

    int local_count = static_cast<int>(local_result.size());

    std::vector<int> recvcounts(size), recv_displs(size);
    MPI_Gather(&local_count, 1, MPI_INT,
               recvcounts.data(), 1, MPI_INT,
               ROOT, MPI_COMM_WORLD);

    int total_count = 0;
    std::vector<int> result;

    if (rank == ROOT) {
        recv_displs[0] = 0;
        for (int i = 1; i < size; ++i)
            recv_displs[i] = recv_displs[i - 1] + recvcounts[i - 1];
        total_count = recv_displs[size - 1] + recvcounts[size - 1];
        result.resize(total_count);
    }

    MPI_Gatherv(
        local_result.data(), local_count, MPI_INT,
        result.data(), recvcounts.data(), recv_displs.data(), MPI_INT,
        ROOT, MPI_COMM_WORLD
    );

    double local_time = t_end - t_start;
    std::vector<double> all_times;
    if (rank == ROOT) all_times.resize(size);

    MPI_Gather(&local_time, 1, MPI_DOUBLE,
               all_times.data(), 1, MPI_DOUBLE,
               ROOT, MPI_COMM_WORLD);

    if (rank == ROOT) {
        printf("Деталі по процесах:\n");
        for (int i = 0; i < size; ++i)
            printf("  Процес %d: отримав %d ел., відфільтрував %d, час = %.6f с\n",
                   i, sendcounts[i], recvcounts[i], all_times[i]);

        printf("\nВідфільтрований масив (%d ел.):\n  ", total_count);
        for (int v : result) printf("%4d", v);
        printf("\n");

        std::vector<int> expected;
        for (int v : global_data)
            if (v > mean) expected.push_back(v);

        std::vector<int> sorted_result = result, sorted_expected = expected;
        std::sort(sorted_result.begin(), sorted_result.end());
        std::sort(sorted_expected.begin(), sorted_expected.end());
        bool ok = (sorted_result == sorted_expected);
        printf("\nПеревірка коректності: %s\n", ok ? "ПРОЙДЕНО" : "ПОМИЛКА");

        double max_time = *std::max_element(all_times.begin(), all_times.end());
        printf("Загальний час (max по процесах): %.6f с\n", max_time);
    }

    MPI_Finalize();
    return 0;
}
