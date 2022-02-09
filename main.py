
import csv
import numpy
import time
import selector as slctr
from sklearn.model_selection import train_test_split
import pandas as pd
import fitnessFUNs

AAA= True
optimizer=[AAA]
datasets=["dermatology"]

NumOfRuns=3
PopulationSize = 40
Iterations= 250

Export=True
ExportToFile="experiment"+time.strftime("%Y-%m-%d-%H-%M-%S")+".csv"
Flag=False

# CSV Header for for the cinvergence 
CnvgHeader1=[]
CnvgHeader2=[]


for l in range(0,Iterations):
	CnvgHeader1.append("Iter"+str(l+1))

for l in range(0,Iterations):
	CnvgHeader2.append("Iter"+str(l+1))


for j in range (0, len(datasets)):        # specfiy the number of the datasets
    for i in range (0, len(optimizer)):
    
        if((optimizer[i]==True)): # start experiment if an optimizer and an objective function is selected
            for k in range (0,NumOfRuns):
                
                #func_details=["costNN",-1,1]
                func_details=fitnessFUNs.getFunctionDetails(0)
                completeData=datasets[j]+".csv"
                x=slctr.selector(i,func_details,PopulationSize,Iterations,completeData)
                  
                if(Export==True):
                    with open(ExportToFile, 'a',newline='\n') as out:
                        writer = csv.writer(out,delimiter=',')
                        if (Flag==False): # just one time to write the header of the CSV file
                            header= numpy.concatenate([["Optimizer","Dataset","objfname","Experiment","startTime","EndTime","ExecutionTime","trainAcc","testAcc"],CnvgHeader1,CnvgHeader1])
                            writer.writerow(header)
                            print(datasets)
                        a=numpy.concatenate([[x.optimizer,datasets[j],x.objfname,k+1,x.startTime,x.endTime,x.executionTime,x.trainAcc,x.testAcc],x.convergence1,x.convergence2])
                        writer.writerow(a)
                    #out.close()
                Flag=True # at least one experiment
                
if (Flag==False): # Faild to run at least one experiment
    print("No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions") 
        
        
