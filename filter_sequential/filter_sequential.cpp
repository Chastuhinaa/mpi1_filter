#include <cstdio>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <numeric>
#include <windows.h>
using namespace std;
using namespace std::chrono;

static const int N    = 1000000;
static const int SEED = 42;

int main() {
    SetConsoleOutputCP(65001);

    vector<int> data(N);
    srand(SEED);
    for (int i = 0; i < N; i++)
        data[i] = (rand() % 200) - 100;

    long long sum = 0;
    for (int x : data) sum += x;
    double mean = static_cast<double>(sum) / N;

    printf("=== Послідовна фільтрація масиву ===\n");
    printf("Розмір масиву      : %d\n", N);
    printf("Середнє арифметичне: %.6f\n", mean);
    printf("Умова фільтрації   : елемент > %.6f\n\n", mean);

    auto t_start = high_resolution_clock::now();

    vector<int> result;
    result.reserve(N);
    for (int x : data)
        if (x > mean) result.push_back(x);

    auto t_end = high_resolution_clock::now();
    duration<double> elapsed = t_end - t_start;

    printf("Знайдено елементів    : %zu\n", result.size());
    printf("Час виконання (послід): %.7f с\n", elapsed.count());

    return 0;
}
