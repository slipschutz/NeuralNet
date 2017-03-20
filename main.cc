




// class Network(object):

//   def __init__(self, sizes):
//   self.num_layers = len(sizes)
//         self.sizes = sizes
//   self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
//   self.weights = [np.random.randn(y, x) 
// 		  for x, y in zip(sizes[:-1], sizes[1:])]


#include <fstream>
#include <iostream>
#include <string>
#include "Network.hh"

using namespace std;

void LoadData(vector <pair <VectorXd,VectorXd> > & data){

  fstream input("./data.txt");
  if (! input.is_open() ){
    cout<<"NO FILE "<<endl;
    return;
  }
  for (int j=0;j<30000;j++){
    VectorXd tempInput(784);
    VectorXd tempOutput(10);
    double v;
    for (int i=0;i<784;i++){
      input>>v;
      tempInput[i]=v;
    }
    string trash;
    input>>trash;
    for (int i=0;i<10;i++){
      input>>v;
      tempOutput[i]=v;
    }
    
    input>>trash;
    data.push_back(make_pair(tempInput,tempOutput));
  }
}



int GetNumber(VectorXd vec){

  double max =-999999;
  int maxIndex=-1;
  for (int i=0;i<vec.size();i++){
    if (vec[i] >max){
      max =vec[i];
      maxIndex=i;
    }
  }
  return maxIndex;
}


int main(int argc, char ** argv){
  srand((unsigned int) time(0));
  
//   MatrixXd mat=MatrixXd::Constant(3,3,1);
//   cout<<mat<<endl;
//   VectorXd vec=VectorXd::Constant(4,10);
//   mat.conservativeResize(mat.rows(), mat.cols()+1);
//   mat.col(mat.cols()-1) = vec;
//   cout<<mat<<endl;

  if (argc != 4){
    cout<<"Give arguments"<<endl;
    return -1;
  }

  int mSize=atoi(argv[1]);
  double eta=atof(argv[2]);
  int epochs = atoi(argv[3]);

  vector < pair <VectorXd,VectorXd> > allData;
  cout<<"Loading data..."<<endl;
  LoadData(allData);
  int trainingSize=floor(0.8*allData.size());

  vector < pair <VectorXd,VectorXd> > trainingData(allData.begin(),allData.begin()+trainingSize);
  vector < pair <VectorXd,VectorXd> > testData(allData.begin()+trainingSize,allData.end());

  cout<<"Done"<<endl;

  


  Network A({783,30,10});
  //data epoch minisize eta
  A.AddInputNode();


  cout<<"Training..."<<endl;
  A.SGD(trainingData,epochs,mSize,eta);
  cout<<"Done "<<endl;


  int totalRight=0;
  for (int i=0;i<testData.size();i++){

    auto ans = A.FeedFoward(testData[i].first);

    if (GetNumber(testData[i].second) == GetNumber(ans) ){
      totalRight++;
    }
  }
  cout<<"total right "<<totalRight<<" "<<totalRight/double(testData.size())<<endl;

  return 0;
}
