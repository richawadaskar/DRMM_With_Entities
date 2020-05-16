/*
 * date 0
 * author eshion
*/

#ifndef _NN4IR_H_
#define _NN4IR_H_

#include<math.h>
#include<iostream>
#include<fstream>
#include<iomanip>
#include<string>
#include<memory>
#include<map>
#include<unordered_map>
#include<unordered_set>
#include<queue>
#include<limits>
#include<algorithm>
#include<thread>
#include<omp.h>
#include<random>
#include"util.h"
#include"Config.h"
#include "Eigen/Dense"


using namespace std;
using namespace Eigen;
#define EIGEN_NO_DBEUG

static omp_lock_t lock;

namespace nsnn4ir{
    enum _enDataType{DROB04,DCLU09,DGOV2,LETORMQ2007};
    enum _enActivationType{SIGMOID, TANH, RELU};
    typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> RMatrixXd;
    typedef Eigen::Matrix<double,1,Eigen::Dynamic,Eigen::RowMajor> VectorXd;
    class Act_Func{
        private:
            _enActivationType m_type;
            VectorXd Sigmoid(const VectorXd & input){
                VectorXd output = VectorXd::Zero(input.size());
                for(int i = 0 ; i < output.size(); ++ i){
                    output[i] = 1.0 / ( 1.0 + exp( -input[i]));
                }
                return std::move(output);
            }
            VectorXd Sigmoid_gd(const VectorXd & input){
                VectorXd vOne = VectorXd::Ones(input.size());
                VectorXd f_gd = (vOne - input).cwiseProduct(input); // curr layer @H'/@H
                return std::move(f_gd);
            }
            VectorXd Tanh(const VectorXd & input){
                VectorXd voutput = VectorXd::Zero(input.size());
                for(int i = 0 ; i < voutput.size(); ++ i){
                    double exp_val = exp( - 2.0 * input[i] );
                    voutput[i] = ( 1.0 - exp_val) / ( 1.0 + exp_val);
                }
                return std::move(voutput);
            }
            VectorXd Tanh_gd(const VectorXd &input){
                VectorXd vOne = VectorXd::Ones(input.size());
                VectorXd f_gd = (vOne - input).cwiseProduct(vOne + input); // curr layer @H'/@H
                return std::move(f_gd);
            }
        public:
            Act_Func(){}
            Act_Func(const _enActivationType & type):m_type(type){}
            VectorXd forward(const VectorXd & input){
                switch(m_type){
                    case _enActivationType::SIGMOID:
                        return Sigmoid(input);
                        break;
                    case _enActivationType::TANH:
                        return Tanh(input);
                        break;
                    case _enActivationType::RELU:
                        break;
                    default:
			            throw MyError("Error: Activation Function type wrong.!");
                        break;
                }
            }
            VectorXd backward(const VectorXd & input){
                switch(m_type){
                    case _enActivationType::SIGMOID:
                        return Sigmoid_gd(input);
                        break;
                    case _enActivationType::TANH:
                        return Tanh_gd(input);
                        break;
                    case _enActivationType::RELU:
                        break;
                    default:
			            throw MyError("Error: Activation Function type wrong.!");
                        break;
                }
            }
    };

    class NN4IR{
        protected:
            _enDataType m_DataType; // Has been used only once
            bool m_CalAllQ = false;
            bool e_CalAllQ = false; // e is for entity
            int m_mini_batch = 20;
            int e_mini_batch = 20; // e is for entitiy
            int m_show_interval = 1000;
            int e_show_interval = 1000;  // e is for entity
            int m_lr_decay_interval = 1000;
            int e_lr_decay_interval = 1000; // e is for entity
            double m_MAX_JUDGEMENT=0;
            double e_MAX_JUDGEMENT=0; // e is for entity
            double m_N = 528155;
            double e_N = 528155; // e is for entity
            double m_DocAvgLen = 247.91;
            double e_DocAvgLen = 247.91; // e is for entity
            double m_CollecLen = 252310006;
            double e_CollecLen = 252310006; // e is for entity
            double m_lr_w1 = 0.02;
            double e_lr_w1 = 0.02;  // e is for entity
            double m_lr_w2 = 0.002;
            double e_lr_w2 = 0.002;  // e is for entity
            // All variables above this can be used as local variables
            // Goal is to define the new variables so that it can be used with entity as well
            WINDEX m_VocabSize; //vector vocab  size
            WINDEX e_VocabSize; //e is for entity
            unordered_map<WINDEX,_stWordInfo> m_vocab;
            unordered_map<WINDEX,_stWordInfo> e_vocab; // e is for entity
            unordered_map<std::string,WINDEX> m_words;
            unordered_map<std::string,WINDEX> e_words; // e is for entity
            unordered_map<WINDEX,VectorXd> m_W;
            unordered_map<WINDEX,VectorXd> e_W; // e is for entity
            unordered_map<QINDEX,_stQRelInfo> m_relinfo;
            unordered_map<QINDEX,_stQRelInfo> e_relinfo; // e is for entity
            unordered_map<QINDEX,vector<string>> m_dataset;
            unordered_map<QINDEX,vector<string>> e_dataset; // e is for entity
            unordered_map<QINDEX,vector<pair<string,string>>> m_QInstance;
            unordered_map<QINDEX,vector<pair<string,string>>> e_QInstance; // e is for entity
            unordered_map<string,_stDocInfo> m_DocCorp;
            unordered_map<string,_stDocInfo> e_DocCorp; // e is for entity
            unordered_map<QINDEX,unordered_map<WINDEX,double>> m_QueryCorp;
            unordered_map<QINDEX,unordered_map<WINDEX,double>> e_QueryCorp; // e is for entity
            unordered_map<QINDEX,vector<WINDEX>> m_seqQueryCorp;
            unordered_map<QINDEX,vector<WINDEX>> e_seqQueryCorp; // e is for entity
            unordered_map<WINDEX,unordered_map<WINDEX,double>> m_QTopKNeighbor;
            unordered_map<WINDEX,unordered_map<WINDEX,double>> e_QTopKNeighbor; // e is for entity

            unordered_map<QINDEX,multimap<double,string,std::greater<double>>> m_RankInfo; // save final ranklist
            unordered_map<QINDEX,multimap<double,string,std::greater<double>>> e_RankInfo; // save final ranklist // e is for entity
            Act_Func m_actfunc;
            Act_Func e_actfunc; // e is for entity
            double NNScore_LCH_IDF(const QINDEX & qindex,const string & currdoc,const vector<RMatrixXd> & vW1,const VectorXd & vW2,const vector<VectorXd> & vW3,vector<RMatrixXd> & vW1_gd,VectorXd & vW2_gd,vector<VectorXd> & vW3_gd,bool bTrain, int wordOrEntity = 0);
        public:
            inline NN4IR(double w1_lr=0.02, double w2_lr=0.002, int minibatch=20, _enActivationType functype=_enActivationType::TANH, bool calallQ=false, int wordOrEntity = 0):m_lr_w1(w1_lr),m_lr_w2(w2_lr), m_actfunc(functype), m_CalAllQ(calallQ){}
            virtual ~NN4IR();
            void RunningMultiThread(int nFold = 5,int maxiter=10, int wordOrEntity = 0);
            void InitWordVec(const std::string &svecfile,bool binary = false, int wordOrEntity = 0); //USES CONDITION
            void InitDocCorp(const std::string & sDocFile,int wordOrEntity = 0); // USES CONDITION
            void InitQueryCorp(const std::string & sQueryFile, int wordOrEntity = 0); // USES CONDITION
            void InitCorpInfo(const std::string & sDFfile,long long docNum = 1247753, int wordOrEntity == 0); // USES CONDITION
            void LoadDataSet(const std::string &sfilenames,int topk = 1000,int pernegative=100,int num_of_instance=100, int wordOrEntity = 0); //USES CONDTION
            void InitGroundTruth(const std::string &sDocFile,const std::string & sIDCGFile,double relevance_level, int wordOrEntity = 0); // USES CONDITION
            void setDataSet(const _enDataType & t, int wordOrEntity = 0){ m_DataType = t;}
            void InitTopKNeiB(int wordOrEntity = 0);
            void EvaluateFile(const std::string & sfilename); // COMMON FUNCTION NO CONDITION NEEDED
            void GetRanklist(const char* sfilename, int wordOrEntity = 0);
        private:
            NN4IR(const NN4IR &) = delete;
            NN4IR & operator=(const NN4IR &) = delete;
            void LoadTermDFCF(const std::string &sfilename="../../data/gov.content.word_df", int wordOrEntity = 0);
            bool Simi_evaluate(const double relevance_level,const QINDEX & qindex,const unordered_map<string,double> & scores,int evaluatenum ,int topk,_stIRResult & eval, int wordOrEntity = 0);
            void kFold(vector<QINDEX> vecQindex,vector<vector<QINDEX>> & vecTrain,int nFold=5,bool shuffle=true, int wordOrEntity = 0);
            void BruceKFold(vector<QINDEX> vecQIndex,vector<vector<QINDEX>> &vecTrain); // COMMON FUNCTION NO CONDITION
    };
}
#endif
