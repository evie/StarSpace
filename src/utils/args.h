/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#pragma once

#include <iostream>
#include <string>
#include <fstream> 

namespace starspace {

class Args {
  public:
    Args();
    std::string trainFile;
    std::string validationFile;
    std::string testFile;
    std::string predictionFile;
    std::string cotapFile;
    std::string model;
    std::string initModel;
    std::string fileFormat;
    std::string label;
    std::string basedoc;
    std::string loss;
    std::string similarity;

    double lr;
    double termLr;
    double norm;
    double margin;
    double initRandSd;
    double p;
    double dropoutLHS;
    double dropoutRHS;
    double wordWeight;
    double posRatio;
    int posPreRange;
    size_t dim;
    int epoch;
    int ws;
    int maxTrainTime;
    int thread;
    int maxNegSamples;
    int negSearchLimit;
    double negSampleRatio;
    int minCount;
    int minCountLabel;
    int bucket;
    int ngrams;
    int trainMode;
    int K;
    bool verbose;
    bool debug;
    bool adagrad;
    bool isTrain;
    bool normalizeText;
    bool saveEveryEpoch;
    bool saveTempModel;
    bool shareEmb;
    bool useWeight;
    bool trainWord;
    bool excludeLHS;
    std::ofstream log_;

    void parseArgs(int, char**);
    void printHelp();
    void printArgs(std::ostream &out);
    void save(std::ostream& out);
    void load(std::istream& in);
    bool isTrue(std::string arg);
};

}
