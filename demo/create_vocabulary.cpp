#include <iostream>
#include <vector>
#include "dirent.h" //ds UNIX only

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
  typedef DBoW2::FBRISK::TDescriptor DescriptorType;
  typedef BRISKVocabulary Vocabulary;
  typedef BRISKDatabase Database;
  std::string descriptor_type_name = "BRISK";
#endif

using namespace DBoW2;
using namespace DUtils;



void loadFeatures(std::vector<std::vector<DescriptorType> >& features, const std::vector<std::string>& images_folders_);
void createVocabulary(const std::vector<std::vector<DescriptorType> >& features);

//! @brief transforms an opencv mat descriptor into a boost dynamic bitset used by dbow2
//! @param[in] descriptor_cv_ the opencv input descriptor
//! @param[out] descriptor_dbow2_ the dbow2 output descriptor
void setDescriptor(const cv::Mat& descriptor_cv_, FBrief::TDescriptor& descriptor_dbow2_);



int32_t main (int32_t argc_, char** argv_) {
  std::vector<std::vector<DescriptorType> > features(0);
  std::vector<std::string> image_folders(0);
  for (uint32_t u = 1; u < static_cast<uint32_t>(argc_); ++u) {
    image_folders.push_back(argv_[u]);
    std::cerr << "loading image folder: " << image_folders.back() << std::endl;
  }
  std::cerr << "press [ENTER] to start processing" << std::endl;
  getchar();

  loadFeatures(features, image_folders);
  createVocabulary(features);
  return 0;
}



void loadFeatures(std::vector<std::vector<DescriptorType> >& features, const std::vector<std::string>& images_folders_) {
  features.clear();
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

  //ds image name filter (only images containing this string are considered)
  const std::string filter = "camera_left";

  //ds stats
  uint64_t total_number_of_extracted_descriptors = 0;

  //ds for each image folder
  for (uint32_t u = 0; u < images_folders_.size(); ++u) {

    //ds parse the image directory
    DIR* handle_directory   = 0;
    struct dirent* iterator = 0;
    if ((handle_directory = opendir(images_folders_[u].c_str()))) {
      while ((iterator = readdir (handle_directory))) {

        //ds buffer file name
        const std::string file_name = iterator->d_name;

        //ds skip paths
        if (file_name == "." || file_name == "..") {
          continue;
        }

        //ds skip if filter doesn't match
        if (file_name.find(filter) == std::string::npos) {
          continue;
        }

        //ds build image string
        const std::string file_path_image = images_folders_[u]+"/"+file_name;

        //ds load image from disk
        const cv::Mat image = cv::imread(file_path_image, CV_LOAD_IMAGE_GRAYSCALE);

        //ds check for invalid image
        if (image.rows == 0 || image.cols == 0) {
          std::cerr << "loadFeatures|WARNING: skipped invalid image: " << file_path_image << std::endl;
          continue;
        }

        //ds detect keypoints and extract descriptors
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        keypoints_detector->detect(image, keypoints);
        descriptor_extractor->compute(image, keypoints, descriptors);
        total_number_of_extracted_descriptors += keypoints.size();
        std::cerr << "image: " << file_path_image
                  << " extracted descriptors: " << keypoints.size()
                  << " (total: " << total_number_of_extracted_descriptors << ")" << std::endl;

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
      closedir(handle_directory);
    } else {
      std::cerr << "loadFeatures|ERROR: unable to access image folder: " << images_folders_[u] << std::endl;
      throw std::runtime_error("invalid image folder");
    }
  }
}

void createVocabulary(const std::vector<std::vector<DescriptorType> >& features) {

  // branching factor and depth levels
  const int k = 10;
  const int L = 6;
  const WeightingType weight = TF_IDF;
  const ScoringType score = L1_NORM;

  Vocabulary voc(k, L, weight, score);

  std::cout << "Creating a " << k << "^" << L << " vocabulary..." << std::endl;
  voc.create(features);
  std::cout << "... done!" << std::endl;

  std::cout << "Vocabulary information: " << voc << std::endl << std::endl;

  // save the vocabulary to disk
  std::cout << std::endl << "Saving vocabulary..." << std::endl;
  voc.save("voc_"+descriptor_type_name+".yml.gz");
  std::cout << "Done" << std::endl;
}

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
