import struct
import sys
from datetime import datetime

def makeBinaryEntityEmbeddingFile():
    time_start = datetime.now()

    f1 = open("entity2vec.bern","r")
    f2 = open("entityEmbedding.bin","wb")
    f3 = open("entity2idProcessed.txt","r")
    numberOfEntities = 75492
    numberOfDimension = 100
    #### Writing 8 bytes of data for ROW and COLUMN
    f2.write(bytes(str(numberOfEntities)+" "+str(numberOfDimension)+"\n","utf-8"))
    #### Read entity from the file
    entity = f3.readline()
    count = 0
    while(entity != ""):
        print(count)
        #Entity has name and number seperated by \t
        entity = entity.split("\t")[0]
        #Write entity to the file in terms of byte and space after it
        f2.write(bytes(entity+" ","utf-8")) # Space after entity
        #Read embedding from the embedding file
        #last is \n character
        embedding  = f1.readline().split("\t")[:-1]
        #Embedding has 100 character floats but this i string here
        for x in embedding:
            #Covert string floats to floats and write in bytes of 4
            f2.write( struct.pack("f",float(x)) )
        entity = f3.readline()
        count += 1

    f1.close()
    f2.close()
    f3.close()
    time_end = datetime.now()
    print("Binary file created",time_start,time_end)

makeBinaryEntityEmbeddingFile()
