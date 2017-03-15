




// class Network(object):

//   def __init__(self, sizes):
//   self.num_layers = len(sizes)
//         self.sizes = sizes
//   self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
//   self.weights = [np.random.randn(y, x) 
// 		  for x, y in zip(sizes[:-1], sizes[1:])]


#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <assert.h>
using namespace std;



class Network{
public:
  Network(vector <int>);
  double Sigmoid(double);
  double Sigmoid_Prime(double);

  vector < vector< vector < double > > > weights;
  vector < vector< double > > biases;
  vector <int> sizes;
  void FeedFoward(vector<double>);
  void PrintWeights();
  vector <double> DotWwithX(int layer,
			    const vector <double> & vec);
  vector <double> DotWwithXTrans(int layer,
			    const vector <double> & vec);

  void BackProp(vector <double> inputs,
		vector <double> trueAnswer,
		vector <vector<double> > & nb,
		vector <vector<vector<double> > >&nw);

  vector<double> cost_derivative(vector<double> output,
				 vector<double> y);
private:
  int rNumberLayers;
  vector <int> rSizes;

};

Network::Network(vector <int> sizes){
  rSizes=sizes;
  //The number of layers will exclude the input layer
  rNumberLayers=sizes.size()-1;

  biases.resize(rNumberLayers);
  for (int i=0;i<rNumberLayers+1;i++){
    //Initialize all the biases to 0
    biases[i].resize(sizes[i],1);
  }

  //Number of layers already excludes the input layer
  weights.resize(rNumberLayers);
  for (int l=0;l<weights.size();l++){
    weights[l].resize(sizes[l]);
    for (int j=0;j<sizes[l];j++){
      weights[l][j].resize(sizes[l+1],1);      
    }
  }
}

double Network::Sigmoid(double z){
  return 1.0/(1.0+exp(-z));
}

double Network::Sigmoid_Prime(double z){
  return Sigmoid(z)*(1-Sigmoid(z));
}

void Network::PrintWeights(){

  for (int l=0;l<weights.size();l++){
    for (int j=0;j<weights[l].size();j++){
      	cout<<"Node in layer "<<l<<" with index"<<j<<" has weight to spot ";
      for (int z=0;z<weights[l][j].size();z++){
	cout<<" z="<<z<<" the value is "<<weights[l][j][z]<<" ";
      }
      cout<<endl<<"-----------------"<<endl;
    }
  }


}

vector<double> Network::DotWwithX(int layer,
				  const vector <double> & vec){
  int l=layer;
  vector <double> nextLayer(rSizes[l+1]);
  for (int j=0;j<rSizes[l];j++){
      //we are taking the value from the input node at index J
      //and we are going to propogate the values to all nodes in the 
      //next layer indexed by z
      for (int z=0;z<weights[l][j].size();z++){
	double num =vec[j]*weights[l][j][z];
	nextLayer[z]+=num;//Sum up the values
      }
    }//end J loop IE done looping over the current set of input nodes

  return nextLayer;
}


vector<double> Network::DotWwithXTrans(int layer,
				       const vector <double> & vec){
  

  int l=layer;
  vector <double> nextLayer(rSizes[l+1]);
  for (int j=0;j<rSizes[l];j++){
      //we are taking the value from the input node at index J
      //and we are going to propogate the values to all nodes in the 
      //next layer indexed by z
      for (int z=0;z<weights[l][j].size();z++){
	double num =vec[j]*weights[l][j][z];
	nextLayer[z]+=num;//Sum up the values
      }
    }//end J loop IE done looping over the current set of input nodes

  return nextLayer;
}





void Network::FeedFoward(vector <double> vec){

  for (int l=0;l<rNumberLayers;l++){
    assert(vec.size()==rSizes[l]);
    //Dot the vector with the weight matrix for the first
    //layer
    vector <double> nextLayer=DotWwithX(l,vec);

    //Now apply the biases to the values and take the sigmoid
    for (int j=0;j<nextLayer.size();j++){
      nextLayer[j]=Sigmoid(nextLayer[j]+biases[l+1][j]);
    }
    vec=nextLayer;
  }//end l loop next iteration will be at next layer of network
  
  for (int i=0;i<vec.size();i++){
    cout<<vec[i]<<endl;
  }
}


void Network::BackProp(vector <double> input,
		       vector <double> trueAnswer,
		       vector<vector< double> > &nabla_b,
		       vector <vector<vector< double > > > &nabla_w){
  

  //need to store the nabla_b's and nabla_w's
  //these need to be the same sizes as the biases and 
  //weight vectors
  //  vector< vector <double> > nabla_b;
  nabla_b.resize(biases.size());
  for (int i=0;i<nabla_b.size();i++){
    nabla_b[i].resize(biases[i].size(),0);
  }

  //  vector<vector<vector< double> > > nabla_w;
  nabla_w.resize(weights.size());
  for (int i=0;i<nabla_w.size();i++){
    nabla_w[i].resize(weights[i].size());
    for (int j=0;j<nabla_w[i].size();j++){
      nabla_w[i][j].resize(weights[i][j].size(),0);
    }
  }
  
  vector<double> activation = input;
  vector< vector <double> > activations;
  activations.push_back(activation);

  vector< vector <double> > zs;
  ///need to do similar feed foward as in the feed foward method
  for (int l=0;l<rNumberLayers;l++){
    ///Z gets set with correct size for next layer in the 
    ///Dot function
    vector <double> z = DotWwithX(l,activation);
    //Activation needs to have the same size as Z the next layers
    //number of nodes
    activation.resize(z.size(),0);
    
    //Now apply the biases to the values and take the sigmoid
    for (int j=0;j<z.size();j++){
      z[j]=z[j]+biases[l][j];
      activation[j]=Sigmoid(z[j]);
    }
    zs.push_back(z);
    activations.push_back(activation);
  }

  assert(activations[activations.size()-1].size()==
	 trueAnswer.size());


//   cout<<"Size of last activation "<<
//     activations[activations.size()-1].size()<<endl;
//   for (int i=0;i<activations[activations.size()-1].size();i++){
//     cout<<activations[activations.size()-1][i]<<endl;
//   }

//   cout<<"Size of last Zs "<<zs[zs.size()-1].size()<<endl;
//   for (int i=0;i<zs[zs.size()-1].size();i++){
//     cout<<zs[zs.size()-1][i]<<endl;
//}


  //done looping through the number of layers
  vector<double> delta = 
    cost_derivative(activations[activations.size()-1],trueAnswer);
  
  for (int i=0;i<delta.size();i++){
    delta[i]=delta[i]*Sigmoid_Prime(zs[zs.size()-1][i]);
  }






  ///Set the vector for the last layer in nabla_b to delta
  nabla_b[nabla_b.size()-1]=delta;

  int s=nabla_w.size()-1;
  vector< vector <double > > temp(nabla_w[s].size());
  for (int i=0;i<nabla_w[s].size();i++){
    temp[i].resize(nabla_w[s][i].size(),0);
  }

  for (int i=0;i<activations[activations.size()-2].size();i++){
    for (int j=0;j<delta.size();j++){
      temp[i][j]=delta[j]*activations[activations.size()-2][i];
    }
  }
  //Set the last layer in nabla_w to temp vector
  nabla_w[s]=temp;


  //now loop over the all the layers 
  //it will go from the last layer  backwards
  //the last layer has already been taken care off above
  //so it will start at the 2nd to last layer
  //  cout<<"-----> "<<rNumberLayers<<endl;

  //  cout<<"noba_w "<<nabla_w.size()<<endl;
  for (int l=1;l<rNumberLayers;l++){

    int theLayer=nabla_w.size()-1-l;
    //    cout<<"the layer is ----asdfaaf>>>>>"<<theLayer<<endl;
    vector<double> z =zs[theLayer];
    vector <double> sp(z.size());
    for (int i=0;i<z.size();i++){
      sp[i]=Sigmoid_Prime(z[i]);
    }

//     for (auto i : delta){
//       cout<<"delta "<<i<<endl;
//     }

    vector <double> delta_temp(weights[theLayer+1].size(),0);

    for (int i=0;i<weights[theLayer+1].size();i++){
      
      //      cout<<"Size for layer "<<theLayer+1<< " "<<weights[theLayer+1].size()<<endl;
      
      for (int j=0;j<weights[theLayer+1][i].size();j++){
	delta_temp[i]+=delta[j]*weights[theLayer+1][i][j];
      }
    }			    

    for (int i=0;i<delta_temp.size();i++){
      delta_temp[i]=delta_temp[i]*sp[i];
    }
    //set delta to delta temp for the next loop
    delta=delta_temp;
    nabla_b[theLayer]=delta;
    

    int size_temp=weights[theLayer].size();
    vector <vector <double > > weight_temp(size_temp);
    //    cout<<"hihihi---->"<<size_temp<<endl;


    for (int i=0;i<size_temp;i++){
      weight_temp[i].resize(weights[theLayer][i].size(),0);
      //      cout<<weight_temp[i].size()<<endl;
    }

    for (int i=0;i<size_temp;i++){
      for (int j=0;j<weight_temp[i].size();j++){
	weight_temp[i][j]=activations[theLayer][i]*delta[j];
      }
    }
    nabla_w[theLayer]=weight_temp;
    

//     for (auto i : delta_temp){
//       cout<<i<<endl;
//     }
//     cout<<"----"<<endl;
//     for (auto i : weight_temp){
//       for (auto j : i){
// 	cout<<j<<" ";
//       }
//       cout<<endl;
//     }
//     cin.get();

  }

}

vector<double> Network::cost_derivative(vector<double> output,
					vector<double> y){
  vector<double> ret;
  for (int i=0;i<output.size();i++){
    ret.push_back(output[i]-y[i]);
  }
  return ret;

}



int main(){

  Network A({3,2,1});
  //  A.PrintWeights();
  //  A.FeedFoward(input);
  vector<double> input={1,2,3};
  vector<double> answer={3};

  vector <vector <double> > nabla_b;
  vector <vector< vector <double> > > nabla_w;
  A.BackProp(input,answer,nabla_b,nabla_w);

  for ( int i=0;i<nabla_b.size();i++){
    cout<<"For Layer "<<i<<endl;
    for (auto j : nabla_b[i]){
      cout<<j <<" ";
    }
    cout<<endl;
  }



  return 0;
}