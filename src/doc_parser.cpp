/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include "doc_parser.h"
#include "utils/normalize.h"
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

#include <boost/algorithm/string.hpp>

using namespace std;

namespace starspace {

LayerDataParser::LayerDataParser(
    shared_ptr<Dictionary> dict,
    shared_ptr<Args> args)
: DataParser(dict, args) {};

bool LayerDataParser::parse(
    string& s,
    ParseResults& ex,
    vector<Base>& feats,
    const string& sep) {

  // split each part into tokens
  vector<string> tokens;
  boost::split(tokens, s, boost::is_any_of(string(sep)));

  int start_idx = 0;
  float ex_weight = 1.0;

  // reset tmpDocInfo
  ex.tmpDocInfo.isNegative = false;
  ex.tmpDocInfo.isApp = false;

  for (int i = start_idx; i < std::min(4, int(tokens.size())); i++) {
      string t = tokens[i];
    // stentence starts with __id__ is sentence id
    if (boost::starts_with(t, "__id__")) {
      ex.tmpDocInfo.id = t.substr(7);
      start_idx = i+1;
    }else if (boost::starts_with(t, "__weight__")) {
      std::size_t pos = t.find(":");
      if (pos != std::string::npos) {
          ex_weight = atof(t.substr(pos + 1).c_str());
      }
      start_idx = i+1;
    } else if (boost::starts_with(t, "__negative__")) {
      ex.tmpDocInfo.isNegative = true;
      start_idx = i+1;
    }
  }

  for (int i = start_idx; i < tokens.size(); i++) {
    string t = tokens[i];
    float weight = 1.0;
    if (args_->useWeight) {
      std::size_t pos = tokens[i].find(":");
      if (pos != std::string::npos) {
        t = tokens[i].substr(0, pos);
        weight = atof(tokens[i].substr(pos + 1).c_str());
      }
    }

    if (args_->normalizeText) {
      normalize_text(t);
    }
    std::size_t pos = t.find(".");
    if (pos != std::string::npos) {
      ex.tmpDocInfo.isApp = true;
    }
    int32_t wid = dict_->getId(t);
    if (wid != -1)  {
      feats.push_back(make_pair(wid, weight * ex_weight));
    }
  }

  if (args_->ngrams > 1) {
    addNgrams(tokens, feats, args_->ngrams);
  }

  return feats.size() > 0;
}

bool LayerDataParser::parse(
    string& line,
    ParseResults& rslt,
    const string& sep) {

  vector<string> parts;
  boost::split(parts, line, boost::is_any_of("\t"));
  int start_idx = 0;

  if (args_->trainMode == 0) {
    // the first part is input features
    parse(parts[start_idx], rslt, rslt.LHSTokens);
    start_idx += 1;
  }
  for (int i = start_idx; i < parts.size(); i++) {
    vector<Base> feats;
    if (parse(parts[i], rslt, feats)) {
      if (rslt.tmpDocInfo.isNegative) {
        rslt.NegFeatures.push_back(feats);
      }else{
        rslt.RHSFeatures.push_back(feats);
	rslt.DocInfos.push_back(rslt.tmpDocInfo);
      }
    }
  }

  bool isValid;
  if (args_->trainMode == 0) {
    isValid = (rslt.LHSTokens.size() > 0) && (rslt.RHSFeatures.size() > 0);
  } else {
    // need to have at least two examples
    isValid = rslt.RHSFeatures.size() > 1;
  }
  //cout << "sample sent size " << rslt.RHSFeatures.size()  << endl;

  return isValid;
}

} // namespace starspace
