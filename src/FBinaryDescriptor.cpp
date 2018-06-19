#include <vector>
#include <string>
#include <sstream>

#include <DVision/DVision.h>

#include "FBinaryDescriptor.h"

using namespace std;

namespace DBoW2 {

// --------------------------------------------------------------------------

void FBinaryDescriptor::meanValue(const std::vector<FBinaryDescriptor::pDescriptor> &descriptors,
    FBinaryDescriptor::TDescriptor &mean)
{
  mean.reset();
  
  if(descriptors.empty()) return;
  
  const int N2 = descriptors.size() / 2;
  const int L = descriptors[0]->size();
  
  vector<int> counters(L, 0);

  vector<FBinaryDescriptor::pDescriptor>::const_iterator it;
  for(it = descriptors.begin(); it != descriptors.end(); ++it)
  {
    const FBinaryDescriptor::TDescriptor &desc = **it;
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
  
double FBinaryDescriptor::distance(const FBinaryDescriptor::TDescriptor &a,
  const FBinaryDescriptor::TDescriptor &b)
{
  return (double)DVision::BRIEF::distance(a, b);
}

// --------------------------------------------------------------------------
  
std::string FBinaryDescriptor::toString(const FBinaryDescriptor::TDescriptor &a)
{
  // from boost::bitset
  string s;
  to_string(a, s); // reversed
  return s;
}

// --------------------------------------------------------------------------
  
void FBinaryDescriptor::fromString(FBinaryDescriptor::TDescriptor &a, const std::string &s)
{
  // from boost::bitset
  stringstream ss(s);
  ss >> a;
}


// --------------------------------------------------------------------------

} // namespace DBoW2

