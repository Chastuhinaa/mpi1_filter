#include <cstdio>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <numeric>
#include <windows.h>
using namespace std;
using namespace std::chrono;

static const int SEED = 42;

int main(int argc, char* argv[]) {
    SetConsoleOutputCP(65001);

    int n = (argc > 1) ? atoi(argv[1]) : 1000000;

    vector<int> data(n);
    srand(SEED);
    for (int i = 0; i < n; i++)
        data[i] = (rand() % 200) - 100;

    long long sum = 0;
    for (int x : data) sum += x;
    double mean = static_cast<double>(sum) / n;

    printf("=== Послідовна фільтрація масиву ===\n");
    printf("Розмір масиву      : %d\n", n);
    printf("Середнє арифметичне: %.6f\n", mean);
    printf("Умова фільтрації   : елемент > %.6f\n\n", mean);

    auto t_start = high_resolution_clock::now();

    vector<int> result;
    result.reserve(n);
    for (int x : data)
        if (x > mean) result.push_back(x);

    auto t_end = high_resolution_clock::now();
    duration<double> elapsed = t_end - t_start;

    printf("Знайдено елементів    : %zu\n", result.size());
    printf("Час виконання (послід): %.7f с\n", elapsed.count());

    return 0;
}
