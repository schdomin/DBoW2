#ifndef __D_T_F_BINARYDESCRIPTOR__
#define __D_T_F_BINARYDESCRIPTOR__

#include <opencv2/core.hpp>
#include <vector>
#include <string>

#include "FClass.h"
#include <DVision/DVision.h>

namespace DBoW2 {

/// Functions to manipulate generic binary descriptors, compared with the hamming distance
class FBinaryDescriptor: protected FClass
{
public:

  //ds we utilize the same bit storage structure as BRIEF descriptors (bitsets)
  typedef DVision::BRIEF::bitset TDescriptor;
  typedef const TDescriptor *pDescriptor;

  /**
   * Calculates the mean value of a set of descriptors
   * @param descriptors
   * @param mean mean descriptor
   */
  static void meanValue(const std::vector<pDescriptor> &descriptors, 
    TDescriptor &mean);
  
  /**
   * Calculates the distance between two descriptors
   * @param a
   * @param b
   * @return distance
   */
  static double distance(const TDescriptor &a, const TDescriptor &b);
  
  /**
   * Returns a string version of the descriptor
   * @param a descriptor
   * @return string version
   */
  static std::string toString(const TDescriptor &a);
  
  /**
   * Returns a descriptor from a string
   * @param a descriptor
   * @param s string version
   */
  static void fromString(TDescriptor &a, const std::string &s);
};

} // namespace DBoW2

#endif

