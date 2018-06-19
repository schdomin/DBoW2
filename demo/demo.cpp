/**
 * File: Demo.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DBoW2
 * License: see the LICENSE.txt file
 */
#include <iostream>
#include <vector>

// DBoW2
#include "DBoW2.h" // defines OrbVocabulary and OrbDatabase

#include <DUtils/DUtils.h>
#include <DVision/DVision.h>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#if CV_MAJOR_VERSION == 2
#elif CV_MAJOR_VERSION == 3
  #include <opencv2/xfeatures2d.hpp>
#else
  #error OpenCV version not supported
#endif

//ds available descriptor types
#define DESCRIPTOR_TYPE_BRIEF 0
#define DESCRIPTOR_TYPE_ORB 1
#define DESCRIPTOR_TYPE_BRISK 2


//ds CHOOSE descriptor type
#define DESCRIPTOR_TYPE DESCRIPTOR_TYPE_BRISK

//ds SET descriptor size
#define DESCRIPTOR_SIZE_BITS 512
#define DESCRIPTOR_SIZE_BYTES DESCRIPTOR_SIZE_BITS/8



//ds determine types
#if DESCRIPTOR_TYPE == DESCRIPTOR_TYPE_BRIEF
  typedef DBoW2::FBrief::TDescriptor DescriptorType;
  typedef BriefVocabulary Vocabulary;
  typedef BriefDatabase Database;
  const std::string descriptor_type_name = "BRIEF";
#elif DESCRIPTOR_TYPE == DESCRIPTOR_TYPE_ORB
  typedef DBoW2::FORB::TDescriptor DescriptorType;
  typedef OrbVocabulary Vocabulary;
  typedef OrbDatabase Database;
  const std::string descriptor_type_name = "ORB";
#elif DESCRIPTOR_TYPE == DESCRIPTOR_TYPE_BRISK
  typedef DBoW2::FBinaryDescriptor::TDescriptor DescriptorType;
  typedef BinaryDescriptorVocabulary Vocabulary;
  typedef BinaryDescriptorDatabase Database;
  std::string descriptor_type_name = "BRISK";
#endif

using namespace DBoW2;
using namespace DUtils;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void loadFeatures(std::vector<std::vector<DescriptorType> >& features);
void createVocabulary(const std::vector<std::vector<DescriptorType> >& features);
void testDatabase(const std::vector<std::vector<DescriptorType> >& features);

//! @brief transforms an opencv mat descriptor into a boost dynamic bitset used by dbow2
//! @param[in] descriptor_cv_ the opencv input descriptor
//! @param[out] descriptor_dbow2_ the dbow2 output descriptor
void setDescriptor(const cv::Mat& descriptor_cv_, FBrief::TDescriptor& descriptor_dbow2_);


// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

// number of training images
const int NIMAGES = 4;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

void wait()
{
  std::cout << std::endl << "Press enter to continue" << std::endl;
  getchar();
}

// ----------------------------------------------------------------------------

int main() {
  std::vector<std::vector<DescriptorType> > features;
  loadFeatures(features);
  createVocabulary(features);
  wait();
  testDatabase(features);
  return 0;
}

// ----------------------------------------------------------------------------

void loadFeatures(std::vector<std::vector<DescriptorType> >& features)
{
  features.clear();
  features.reserve(NIMAGES);
  cv::Ptr<cv::FeatureDetector> keypoints_detector;
  cv::Ptr<cv::DescriptorExtractor> descriptor_extractor;

#if DESCRIPTOR_TYPE == DESCRIPTOR_TYPE_BRIEF
  keypoints_detector   = cv::FastFeatureDetector::create(10);
  descriptor_extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(DESCRIPTOR_SIZE_BYTES);
  std::cout << "Extracting BRIEF features..." << std::endl;
#elif DESCRIPTOR_TYPE == DESCRIPTOR_TYPE_ORB
  keypoints_detector   = cv::ORB::create(1000);
  descriptor_extractor = cv::ORB::create(1000);
  std::cout << "Extracting ORB features..." << std::endl;
#elif DESCRIPTOR_TYPE == DESCRIPTOR_TYPE_BRISK
  keypoints_detector   = cv::BRISK::create(25);
  descriptor_extractor = cv::BRISK::create(25);
  std::cout << "Extracting BRISK features..." << std::endl;
#endif

  for(int i = 0; i < NIMAGES; ++i)
  {
    std::stringstream ss;
    ss << "images/image" << i << ".png";

    cv::Mat image = cv::imread(ss.str(), 0);
    cv::Mat mask;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    //ds detect keypoints and extract descriptors
    keypoints_detector->detect(image, keypoints);
    descriptor_extractor->compute(image, keypoints, descriptors);
    std::cerr << "image: " << i << " extracted descriptors: " << keypoints.size() << std::endl;

    //ds convert descriptors to BoW format
    std::vector<DescriptorType> descriptors_bow(descriptors.rows);
    for(int32_t u = 0; u < descriptors.rows; ++u) {
#if DESCRIPTOR_TYPE == DESCRIPTOR_TYPE_BRIEF or DESCRIPTOR_TYPE == DESCRIPTOR_TYPE_BRISK
      setDescriptor(descriptors.row(u), descriptors_bow[u]);
#elif DESCRIPTOR_TYPE == DESCRIPTOR_TYPE_ORB
      descriptors_bow[u] = descriptors.row(u);
#endif
    }
    features.push_back(descriptors_bow);
  }
}

// ----------------------------------------------------------------------------

void createVocabulary(const std::vector<std::vector<DescriptorType> >& features)
{
  // branching factor and depth levels 
  const int k = 9;
  const int L = 3;
  const WeightingType weight = TF_IDF;
  const ScoringType score = L1_NORM;

  Vocabulary voc(k, L, weight, score);

  std::cout << "Creating a small " << k << "^" << L << " vocabulary..." << std::endl;
  voc.create(features);
  std::cout << "... done!" << std::endl;

  std::cout << "Vocabulary information: " << std::endl
  << voc << std::endl << std::endl;

  // lets do something with this vocabulary
  std::cout << "Matching images against themselves (0 low, 1 high): " << std::endl;
  BowVector v1, v2;
  for(int i = 0; i < NIMAGES; i++)
  {
    voc.transform(features[i], v1);
    for(int j = 0; j < NIMAGES; j++)
    {
      voc.transform(features[j], v2);
      
      double score = voc.score(v1, v2);
      std::cout << "Image " << i << " vs Image " << j << ": " << score << std::endl;
    }
  }

  // save the vocabulary to disk
  std::cout << std::endl << "Saving vocabulary..." << std::endl;
  voc.save("small_voc_"+descriptor_type_name+".yml.gz");
  std::cout << "Done" << std::endl;
}

// ----------------------------------------------------------------------------

void testDatabase(const std::vector<std::vector<DescriptorType> >& features)
{
  std::cout << "Creating a small database..." << std::endl;

  // load the vocabulary from disk
  Vocabulary voc("small_voc_"+descriptor_type_name+".yml.gz");
  Database db(voc, false, 0); // false = do not use direct index
  // (so ignore the last param)
  // The direct index is useful if we want to retrieve the features that 
  // belong to some vocabulary node.
  // db creates a copy of the vocabulary, we may get rid of "voc" now

  // add images to the database
  for(int i = 0; i < NIMAGES; i++)
  {
    db.add(features[i]);
  }

  std::cout << "... done!" << std::endl;

  std::cout << "Database information: " << std::endl << db << std::endl;

  // and query the database
  std::cout << "Querying the database: " << std::endl;

  QueryResults ret;
  for(int i = 0; i < NIMAGES; i++)
  {
    db.query(features[i], ret, 4);

    // ret[0] is always the same image in this case, because we added it to the 
    // database. ret[1] is the second best match.

    std::cout << "Searching for Image " << i << ". " << ret << std::endl;
  }

  std::cout << std::endl;

  // we can save the database. The created file includes the vocabulary
  // and the entries added
  std::cout << "Saving database..." << std::endl;
  db.save("small_db_"+descriptor_type_name+".yml.gz");
  std::cout << "... done!" << std::endl;
  
  // once saved, we can load it again  
  std::cout << "Retrieving database once again..." << std::endl;
  Database db2("small_db_"+descriptor_type_name+".yml.gz");
  std::cout << "... done! This is: " << std::endl << db2 << std::endl;
}

// ----------------------------------------------------------------------------

void setDescriptor(const cv::Mat& descriptor_cv_, FBrief::TDescriptor& descriptor_dbow2_) {
  FBrief::TDescriptor bit_buffer(DESCRIPTOR_SIZE_BITS);

  //ds loop over all bytes
  for (uint32_t u = 0; u < DESCRIPTOR_SIZE_BYTES; ++u) {

    //ds get minimal datafrom cv::mat
    const uchar byte_value = descriptor_cv_.at<uchar>(u);

    //ds get bitstring
    for(uint8_t v = 0; v < 8; ++v) {
      bit_buffer[u*8+v] = (byte_value >> v) & 1;
    }
  }
  descriptor_dbow2_ = bit_buffer;
}
