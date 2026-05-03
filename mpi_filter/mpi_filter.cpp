#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <algorithm>
#include <numeric>
#include <windows.h>

static const int N = 1000000;
static const int SEED = 42;
static const int ROOT = 0;

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    SetConsoleOutputCP(65001);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = (argc > 1) ? atoi(argv[1]) : N;
    if (n < size) n = size;

    int base = n / size;
    int extra = n % size;

    std::vector<int> sendcounts(size), displs(size);
    for (int i = 0; i < size; ++i) {
        sendcounts[i] = base + (i < extra ? 1 : 0);
        displs[i] = (i == 0) ? 0 : displs[i - 1] + sendcounts[i - 1];
    }

    int local_n = sendcounts[rank];
    std::vector<int> local_data(local_n);

    std::vector<int> global_data;
    double mean = 0.0;

    if (rank == ROOT) {
        global_data.resize(n);
        srand(SEED);
        for (int i = 0; i < n; ++i)
            global_data[i] = (rand() % 200) - 100;

        long long sum = std::accumulate(global_data.begin(), global_data.end(), 0LL);
        mean = static_cast<double>(sum) / n;

        printf("=== Паралельна фільтрація масиву (MPI) ===\n");
        printf("Кількість процесів : %d\n", size);
        printf("Розмір масиву      : %d\n", n);
        printf("Середнє арифметичне: %.6f\n", mean);
        printf("Умова фільтрації   : елемент > %.6f\n\n", mean);
    }

    MPI_Bcast(&mean, 1, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    MPI_Scatterv(
        global_data.data(), sendcounts.data(), displs.data(), MPI_INT,
        local_data.data(), local_n, MPI_INT,
        ROOT, MPI_COMM_WORLD
    );

    std::vector<int> local_result;
    local_result.reserve(local_n);
    for (int v : local_data)
        if (v > mean) local_result.push_back(v);

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
    else {
     
        std::fill(recvcounts.begin(), recvcounts.end(), 0);
        std::fill(recv_displs.begin(), recv_displs.end(), 0);
    }

    MPI_Gatherv(
        local_result.data(), local_count, MPI_INT,
        result.data(), recvcounts.data(), recv_displs.data(), MPI_INT,
        ROOT, MPI_COMM_WORLD
    );

    double t_end = MPI_Wtime();


    double local_time = t_end - t_start;
    std::vector<double> all_times;
    if (rank == ROOT) all_times.resize(size);
    MPI_Gather(&local_time, 1, MPI_DOUBLE,
        all_times.data(), 1, MPI_DOUBLE,
        ROOT, MPI_COMM_WORLD);

    if (rank == ROOT) {
        printf("Знайдено елементів : %d\n", total_count);
        printf("Відфільтрований масив (перші 20): ");
        int show = (total_count < 20) ? total_count : 20;
        for (int i = 0; i < show; ++i) printf("%d ", result[i]);
        printf("\n\n");

        printf("Деталі по процесах:\n");
        for (int i = 0; i < size; ++i)
            printf("  Процес %d: отримав %d ел., відфільтрував %d, час = %.6f с\n",
                i, sendcounts[i], recvcounts[i], all_times[i]);

        double parallel_time = *std::max_element(all_times.begin(), all_times.end());
        printf("\nЧас виконання (MPI): %.7f с\n", parallel_time);

        std::vector<int> expected;
        expected.reserve(n);
        for (int v : global_data)
            if (v > mean) expected.push_back(v);

        std::vector<int> sr = result, se = expected;
        std::sort(sr.begin(), sr.end());
        std::sort(se.begin(), se.end());
        printf("Перевірка коректності: %s\n", (sr == se) ? "ПРОЙДЕНО" : "ПОМИЛКА");
    }

    MPI_Finalize();
    return 0;
}