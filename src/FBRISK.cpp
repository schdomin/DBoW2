#include <vector>
#include <string>
#include <sstream>

#include <DVision/DVision.h>
#include "FBRISK.h"

using namespace std;

namespace DBoW2 {

// --------------------------------------------------------------------------

void FBRISK::meanValue(const std::vector<FBRISK::pDescriptor> &descriptors,
    FBRISK::TDescriptor &mean)
{
  mean.reset();
  
  if(descriptors.empty()) return;
  
  const int N2 = descriptors.size() / 2;
  const int L = descriptors[0]->size();
  
  vector<int> counters(L, 0);

  vector<FBRISK::pDescriptor>::const_iterator it;
  for(it = descriptors.begin(); it != descriptors.end(); ++it)
  {
    const FBRISK::TDescriptor &desc = **it;
    for(int i = 0; i < L; ++i)
    {
      if(desc[i]) counters[i]++;
    }
  }
  
  for(int i = 0; i < L; ++i)
  {
    if(counters[i] > N2) mean.set(i);
  }
  
}

// --------------------------------------------------------------------------
  
double FBRISK::distance(const FBRISK::TDescriptor &a,
  const FBRISK::TDescriptor &b)
{
  return (double)DVision::BRIEF::distance(a, b);
}

// --------------------------------------------------------------------------
  
std::string FBRISK::toString(const FBRISK::TDescriptor &a)
{
  // from boost::bitset
  string s;
  to_string(a, s); // reversed
  return s;
}

// --------------------------------------------------------------------------
  
void FBRISK::fromString(FBRISK::TDescriptor &a, const std::string &s)
{
  // from boost::bitset
  stringstream ss(s);
  ss >> a;
}

// --------------------------------------------------------------------------

void FBRISK::toMat64F(const std::vector<TDescriptor> &descriptors,
  cv::Mat &mat)
{
  if(descriptors.empty())
  {
    mat.release();
    return;
  }
  
  const int N = descriptors.size();
  const int L = descriptors[0].size();
  
  mat.create(N, L, CV_64F);
  
  for(int i = 0; i < N; ++i)
  {
    const TDescriptor& desc = descriptors[i];
    float *p = mat.ptr<float>(i);
    for(int j = 0; j < L; ++j, ++p)
    {
      *p = (desc[j] ? 1 : 0);
    }
  } 
}

// --------------------------------------------------------------------------

} // namespace DBoW2

