#include <stdio.h>
#include <string>
#include <iostream>
using namespace std;

int main(){
  long long row,col;
	std::string text;
	std::string lowtext;
  long count = 0;

  FILE *f = fopen("entityEmbedding.bin","rb");
  fscanf(f,"%lld",&row);
  fscanf(f,"%lld",&col);
	printf("Row is %lld\n",row);
  printf("Col is %lld\n",col);
	printf("\n\n");
  float buf[col];

  //return 0 ;

  for(int i = 0 ;i < row; ++i){
		count += 1;
		std::string text = "";
		std::string lowtext="";
    while(1){
        char c = fgetc(f);
        if(feof(f) || c == ' ') break;
        text += c;
    }

    std::cout<<text<<"\n";

    //vector<float> currv(col,0.0);
    fread(buf,sizeof(float),col,f);
    for(int j = 0 ; j < col; ++ j)  printf("%f\t",buf[j]);
    printf("\n");
    //return 0;
  }

	printf("%ld\n", count);
	printf("%lld\n",row);
	printf("%lld",col);
  fclose(f);
}
