//
// Created by Oliver_Sun on 2020/3/17.
//
#define TEST
#define _TIME_TEST

#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#include <time.h>
#include <cmath>
#include <assert.h>
#include <string.h>
#include <numeric>

using namespace std;

class LogisticRegression {
private:
    // 最大迭代次数
    static const int maxIterTimes = 2;
    // 学习率
    float learningRate = 0.0;
    // 特征矩阵
    vector<float> coefficients_;
public:
    void Fit(vector<vector<float>>& trainFeatures, const vector<int>& labels) {
        coefficients_.resize(trainFeatures[0].size(), 1.0);
        for (int iter = 0; iter < maxIterTimes; ++iter) {
            for (int index = 0; index < trainFeatures.size(); ++index) {
                learningRate = 1.0 / (10.0 +  iter + index) + 0.01;
                float result = sigmoid(dot(trainFeatures[index], coefficients_));
                float error = labels[index] - result;
                for (int j = 0; j < coefficients_.size(); j += 1) {
                    coefficients_[j] += learningRate * error * trainFeatures[index][j];
                }
            }
        }
    }
    //static inline
    float dot(const vector<float>& a, const vector<float>& b) {
        assert(a.size() == b.size());
        float sum = 0.0;
        for (int i = 0; i < a.size(); ++i) {
            sum += (a[i] * b[i]);
        }
        return sum;
    }

    static inline float sigmoid(float x) {
        return 1.0 / (1.0 + exp(-x));
    }

    vector<int> Predict(vector<vector<float>>& testFeatures) {
//        reshape(testFeatures);
        vector<int> predictVal(testFeatures.size());
        for (int i = 0; i < testFeatures.size(); ++i) {
            float predict = 0.0;
            for (int j = 0; j < coefficients_.size(); ++j) {
                predict += (coefficients_[j] * testFeatures[i][j]);
            }
            predictVal[i] = (sigmoid(predict) >= 0.5 ? 1 : 0);
        }
        return predictVal;
    }
};

bool loadTrainData(const char *trainFile, vector<vector<float>> &features, vector<int> &labels) {
    char *file = NULL;
    int fd = open(trainFile, O_RDONLY);
    long long size = lseek(fd, 0, SEEK_END);
    file = (char *) mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    char tmp[10];
    vector<float> feature;
    int i = 0, j = 0;
    while (j < size) {
        if (file[j] == '\n') {
            strncpy(tmp,file + i, j - i);
            int label = atoi(tmp);
            labels.push_back(label);
            feature.push_back(1.0);
            features.push_back(feature);
            feature.clear();
            i = j + 1;
        } else if (file[j] == ',') {
            strncpy(tmp, file + i, j - i);
            float f = atof(tmp);
            feature.push_back(f);
            i = j + 1;
        }
        ++j;
    }
    return true;
}

bool loadTestData(const char *testFile, vector<vector<float>> &features) {
    char *file = NULL;
    int fd = open(testFile, O_RDONLY);
    long size = lseek(fd,0, SEEK_END);
    file = (char *) mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    char tmp[10];
    vector<float> feature;
    int i = 0, j = 0;
    while (j < size) {
        if (file[j] == '\n') {
            strncpy(tmp, file + i, j - i);
            float f = atof(tmp);
            feature.push_back(f);
            feature.push_back(1.0);
            features.push_back(feature);
            feature.clear();
            i = j + 1;
        } else if (file[j] == ',') {
            strncpy(tmp, file + i, j - i);
            float f = atof(tmp);
            feature.push_back(f);
            i = j + 1;
        }
        ++j;
    }
    return true;
}

bool loadAnswerData(const char *answer, vector<int>& labels) {
    char *file = NULL;
    int fd = open(answer, O_RDONLY);
    long size = lseek(fd, 0, SEEK_END);
    file = (char *) mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);
    char tmp[10];
    int i = 0, j = 0;
    while (j < size) {
        if (file[j] == '\n') {
            strncpy(tmp, file + i, j - i);
            int f = atoi(tmp);
            labels.push_back(f);
            i = j + 1;
        }
        ++j;
    }
    return true;
}

void storeResult(const char* predictFile, const vector<int>& predictVal) {
    const int len = (predictVal.size()) << 1;
    char* result = new char[len];
    for (int i = 0; i <= len; i += 2) {
        result[i] = (predictVal[i >> 1] == 0 ? '0' : '1');
        result[i + 1] = '\n';
    }
    int fd = open(predictFile, O_RDWR | O_CREAT, S_IRUSR | S_IRGRP | S_IROTH);
    lseek(fd, len - 1, SEEK_SET);
    write(fd, "", 1);
    void* p = mmap(NULL, len, PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    fd = -1;
    memcpy(p, result, len);
    munmap(p, len);
    p = NULL;
    delete[] result;
    return;
}

int main() {
#ifdef _TIME_TEST
    clock_t start = clock();
#endif

#ifdef TEST
    const char* trainFile = "../data/train_data.txt";
    const char* testFile = "../data/test_data.txt";
    const char* answer = "../data/answer.txt";
#else
    const char* trainFile = "/data/train_data.txt";
    const char* testFile = "/data/test_data.txt";
    const char* predictFile = "/projects/student/result.txt";
#endif

    vector<vector<float>> trainFeatures, testFeatures;
    vector<int> trainLabels, testLabels;
    vector<int> predict;

    LogisticRegression lr;

    loadTrainData(trainFile, trainFeatures, trainLabels);
    loadTestData(testFile, testFeatures);

    lr.Fit(trainFeatures, trainLabels);

    predict = lr.Predict(testFeatures);

#ifndef TEST
    storeResult(predictFile, predict);
#endif

#ifdef TEST
    loadAnswerData(answer, testLabels);
    int correctCount = 0;
    for (int j = 0; j < predict.size(); j++) {
        if (j < testLabels.size()) {
            if (predict[j] == testLabels[j]) {
                correctCount++;
            }
        } else {
            cout << "answer size less than the real predicted value" << endl;
        }
    }
    float accurate = ((float) correctCount) / testLabels.size();
    cout << "the prediction accuracy is " << accurate << endl;
#endif

#ifdef _TIME_TEST
    clock_t ends = clock();
    cout <<"Running Time : "<<(float)(ends - start)/ CLOCKS_PER_SEC << endl;
#endif
    return 0;
}