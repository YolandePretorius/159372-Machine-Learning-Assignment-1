import ga#parameters for the ga function: (stringLength,fitnessFunction,nEpochs,populationSize=100,mutationProb=-1,crossover='un',nElite=4,tournament=True)# g = ga.ga(100,'fF.fourpeaks',1000,100,0.01,'sp',4,False)# g.runGA()# #g = ga.ga(39,'mlpf.mlpfit',10,40,1,'un',10,True)print("running GA")g.runGA()