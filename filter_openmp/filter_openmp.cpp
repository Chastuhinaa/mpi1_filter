#include <omp.h>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <numeric>
#include <algorithm>
#include <windows.h>

static const int N    = 1000000;
static const int SEED = 42;

int main(int argc, char* argv[])
{
    SetConsoleOutputCP(65001);

    int n = (argc > 1) ? atoi(argv[1]) : N;
    int num_threads = (argc > 2) ? atoi(argv[2]) : omp_get_max_threads();
    omp_set_num_threads(num_threads);

    std::vector<int> data(n);
    srand(SEED);
    for (int i = 0; i < n; i++)
        data[i] = (rand() % 200) - 100;

    long long sum = 0;
    for (int x : data) sum += x;
    double mean = static_cast<double>(sum) / n;

    printf("=== Паралельна фільтрація масиву (OpenMP) ===\n");
    printf("Кількість потоків   : %d\n", num_threads);
    printf("Розмір масиву       : %d\n", n);
    printf("Середнє арифметичне : %.6f\n", mean);
    printf("Умова фільтрації    : елемент > %.6f\n\n", mean);

    std::vector<int> result;
    result.reserve(n / 2);
    std::vector<int> thread_counts(num_threads, 0);

    double t_start = omp_get_wtime();

    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
    
        std::vector<int> local;

        #pragma omp for schedule(static)
        for (int i = 0; i < n; i++)
            if (data[i] > mean)
                local.push_back(data[i]);

        thread_counts[tid] = static_cast<int>(local.size());

        #pragma omp critical
        result.insert(result.end(), local.begin(), local.end());
    }

    double t_end = omp_get_wtime();
    double elapsed = t_end - t_start;

    printf("Знайдено елементів  : %zu\n", result.size());
    printf("Відфільтрований масив (перші 20): ");
    int show = (static_cast<int>(result.size()) < 20) ? static_cast<int>(result.size()) : 20;
    for (int i = 0; i < show; i++) printf("%d ", result[i]);
    printf("\n\n");

    printf("Деталі по потоках:\n");
    for (int i = 0; i < num_threads; i++)
        printf("  Потік %d: відфільтрував %d ел.\n", i, thread_counts[i]);

    printf("\nЧас виконання (OpenMP): %.7f с\n", elapsed);

    std::vector<int> expected;
    expected.reserve(n / 2);
    for (int x : data)
        if (x > mean) expected.push_back(x);

    std::vector<int> sr = result, se = expected;
    std::sort(sr.begin(), sr.end());
    std::sort(se.begin(), se.end());
    printf("Перевірка коректності: %s\n", (sr == se) ? "ПРОЙДЕНО" : "ПОМИЛКА");

    return 0;
}
