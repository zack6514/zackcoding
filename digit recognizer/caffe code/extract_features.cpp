#include <stdio.h>  // for snprintf
#include <string>
#include <vector>
#include <fstream>

#include "boost/algorithm/string.hpp"
#include "google/protobuf/text_format.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/vision_layers.hpp"

using caffe::Blob;
using caffe::Caffe;
using caffe::Datum;
using caffe::Net;
using boost::shared_ptr;
using std::string;
namespace db = caffe::db;

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv);

int main(int argc, char** argv) {
  return feature_extraction_pipeline<float>(argc, argv);
//  return feature_extraction_pipeline<double>(argc, argv);
}

template<typename Dtype>
int feature_extraction_pipeline(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  const int num_required_args = 7;     /// the parameters must be not less 7
  if (argc < num_required_args) {
    LOG(ERROR)<<
    "This program takes in a trained network and an input data layer, and then"
    " extract features of the input data produced by the net.\n"
    "Usage: extract_features  pretrained_net_param"
    "  feature_extraction_proto_file  extract_feature_blob_name1[,name2,...]"
    "  save_feature_dataset_name1[,name2,...]  num_mini_batches  db_type"
    "  [CPU/GPU] [DEVICE_ID=0]\n"
    "Note: you can extract multiple features in one pass by specifying"
    " multiple feature blob names and dataset names seperated by ','."
    " The names cannot contain white space characters and the number of blobs"
    " and datasets must be equal.";
    return 1;
  }
  int arg_pos = num_required_args;     //the necessary nums of parameters

  arg_pos = num_required_args;
  if (argc > arg_pos && strcmp(argv[arg_pos], "GPU") == 0) {          // whether use GPU------ -gpu 0
    LOG(ERROR)<< "Using GPU";
    uint device_id = 0;
    if (argc > arg_pos + 1) {
      device_id = atoi(argv[arg_pos + 1]);
      CHECK_GE(device_id, 0);
    }
    LOG(ERROR) << "Using Device_id=" << device_id;
    Caffe::SetDevice(device_id);
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  arg_pos = 0;  // the name of the executable
  std::string pretrained_binary_proto(argv[++arg_pos]);      // the mode had been trained

  // Expected prototxt contains at least one data layer such as
  //  the layer data_layer_name and one feature blob such as the
  //  fc7 top blob to extract features.
  /*
   layers {
     name: "data_layer_name"
     type: DATA
     data_param {
       source: "/path/to/your/images/to/extract/feature/images_leveldb"
       mean_file: "/path/to/your/image_mean.binaryproto"
       batch_size: 128
       crop_size: 227
       mirror: false
     }
     top: "data_blob_name"
     top: "label_blob_name"
   }
   layers {
     name: "drop7"
     type: DROPOUT
     dropout_param {
       dropout_ratio: 0.5
     }
     bottom: "fc7"
     top: "fc7"
   }
   */
  std::string feature_extraction_proto(argv[++arg_pos]);    // get the net structure
  shared_ptr<Net<Dtype> > feature_extraction_net(
      new Net<Dtype>(feature_extraction_proto, caffe::TEST));               //new net object  and set each layers------feature_extraction_net
  feature_extraction_net->CopyTrainedLayersFrom(pretrained_binary_proto);           // init the weights

  std::string extract_feature_blob_names(argv[++arg_pos]);          //exact which blob's feature
  std::vector<std::string> blob_names;
  boost::split(blob_names, extract_feature_blob_names, boost::is_any_of(","));   //you can exact many blobs' features and to store them in different dirname

  std::string save_feature_dataset_names(argv[++arg_pos]);   // to store the features
  std::vector<std::string> dataset_names;
  boost::split(dataset_names, save_feature_dataset_names,         // each dataset_names to store one blob's feature
               boost::is_any_of(","));
  CHECK_EQ(blob_names.size(), dataset_names.size()) <<
      " the number of blob names and dataset names must be equal";
  size_t num_features = blob_names.size();     // how many features you exact

  for (size_t i = 0; i < num_features; i++) {
    CHECK(feature_extraction_net->has_blob(blob_names[i]))
        << "Unknown feature blob name " << blob_names[i]
        << " in the network " << feature_extraction_proto;
  }

  int num_mini_batches = atoi(argv[++arg_pos]);            // each exact num_mini_batches of images

  // init the DB and Transaction for all blobs you want to extract features
  std::vector<shared_ptr<db::DB> > feature_dbs;               // new DB object, is a vector  maybe has many blogs' feature
  std::vector<shared_ptr<db::Transaction> > txns;            // new Transaction object, is a vectore maybe has many blob's feature


  // edit by Zack
   //std::string strfile = "/home/hadoop/caffe/textileImage/features/probTest";
  std::string strfile = argv[argc-1];
  std::vector<std::ofstream*> vec(num_features, 0);

  const char* db_type = argv[++arg_pos];                  //the data to store style == lmdb
  for (size_t i = 0; i < num_features; ++i) {
    LOG(INFO)<< "Opening dataset " << dataset_names[i];               // dataset_name[i] to store the feature which type is lmdb
    shared_ptr<db::DB> db(db::GetDB(db_type));             // the type of the db
    db->Open(dataset_names.at(i), db::NEW);          // open the dir to store the feature
    feature_dbs.push_back(db);             // put the db to the vector
    shared_ptr<db::Transaction> txn(db->NewTransaction());     // the transaction to the db
    txns.push_back(txn);                // put the transaction to the vector

// edit by Zack

    std::stringstream ss;
    ss.clear();
    string index;
    ss << i;
    ss >> index;
    std::string str = strfile + index + ".txt";
    vec[i] = new std::ofstream(str.c_str());
  }

  LOG(ERROR)<< "Extacting Features";

  Datum datum;
  const int kMaxKeyStrLength = 100;
  char key_str[kMaxKeyStrLength];      // to store the key
  std::vector<Blob<float>*> input_vec;
  std::vector<int> image_indices(num_features, 0);   /// how many blogs' feature you exact


  for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index) {
    feature_extraction_net->Forward(input_vec);
    for (int i = 0; i < num_features; ++i) {    // to exact the blobs' name  maybe fc7 fc8
      const shared_ptr<Blob<Dtype> > feature_blob = feature_extraction_net
          ->blob_by_name(blob_names[i]);
      int batch_size = feature_blob->num();     // the nums of images-------batch size
      int dim_features = feature_blob->count() / batch_size;    // this dim of this feature of each image in this blob
      const Dtype* feature_blob_data;   // float is the features
      for (int n = 0; n < batch_size; ++n) {
        datum.set_height(feature_blob->height());     // set the height
        datum.set_width(feature_blob->width());     // set the width
        datum.set_channels(feature_blob->channels());    // set the channel
        datum.clear_data();               // clear data
        datum.clear_float_data();        // clear float_data
        feature_blob_data = feature_blob->cpu_data() +
            feature_blob->offset(n);    //the features of  which image
        for (int d = 0; d < dim_features; ++d) {
          datum.add_float_data(feature_blob_data[d]);
          (*vec[i]) << feature_blob_data[d] << " ";          // save the features
        }
        (*vec[i]) << std::endl;
        //LOG(ERROR)<< "dim" << dim_features;
        int length = snprintf(key_str, kMaxKeyStrLength, "%010d",
            image_indices[i]);       // key  di ji ge tupian
        string out;
        CHECK(datum.SerializeToString(&out));    // serialize to string
        txns.at(i)->Put(std::string(key_str, length), out);       // put to transaction
        ++image_indices[i];       // key++
        if (image_indices[i] % 1000 == 0) {    // when it reach to 1000 ,we commit it
          txns.at(i)->Commit();
          txns.at(i).reset(feature_dbs.at(i)->NewTransaction());
          LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
              " query images for feature blob " << blob_names[i];
        }
      }  // for (int n = 0; n < batch_size; ++n)
    }  // for (int i = 0; i < num_features; ++i)
  }  // for (int batch_index = 0; batch_index < num_mini_batches; ++batch_index)
  // write the last batch
  for (int i = 0; i < num_features; ++i) {
    if (image_indices[i] % 1000 != 0) {     // commit the last path images
      txns.at(i)->Commit();
    }
    // edit by Zack
      vec[i]->close();
      delete vec[i];

    LOG(ERROR)<< "Extracted features of " << image_indices[i] <<
        " query images for feature blob " << blob_names[i];
    feature_dbs.at(i)->Close();
  }

  LOG(ERROR)<< "Successfully extracted the features!";
  return 0;
}

