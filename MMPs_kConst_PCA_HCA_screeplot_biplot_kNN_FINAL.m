%% Data Loading and Pre-Processing
clc
clear
set(0,'DefaultFigureWindowStyle','docked')
close all

training_data = readtable('ExpFit_logKConstMMPs.xlsx');
XTrain = table2array(training_data(1:15, 2:12));
YTrain= table2array(training_data(1:15,1));


proteases=categorical(YTrain);
labels = YTrain;
C = categories(proteases); %class names


%% HCA
hca =linkage (XTrain, 'average', 'cityblock');
figure ('Name', 'HCA');
dendrogram(hca,'Labels',labels)

%% PCA
% This code returns four outputs: coeff, scoreTrain, explained, and mu. 
%Use explained (percentage of total variance explained) to find the number 
% of components required to explain at least 95% variability. Use coeff 
%(principal component coefficients) and mu (estimated means of XTrain) to 
%apply the PCA to a test data set. Use scoreTrain (principal component
%scores) instead of XTrain when you train a model.
 
[coeff,score,latent,tsquared,explained, mu] = pca(XTrain);

idx = find(cumsum(explained)>97,1)

figure('Name','Scree Plot')
plot(latent,'-.o') %Create scree plot from eigenvalues contained in the latent output 
ylabel('Eigenvalue','FontWeight', 'bold')
xlabel('Principle Component Number','FontWeight', 'bold')
title('Scree Plot')

figure('Name','2D Score Plot')

p =gscatter(score(:, 1), score(:, 2), YTrain)

PC1=explained(1);
PC2=explained(2);

xlabel('Component 1 (82.6%)') % how can I add percent ('%s %', explained(1))')
ylabel('Component 2 (12.8%)')


%Divide scores by protease
score1=score(1:3,1:2);
x1=score1(:,1); y1=score1(:,2);
score2=score(4:6,1:2);
x2=score2(:,1); y2=score2(:,2);
score3=score(7:9,1:2);
x3=score3(:,1); y3=score3(:,2);
score4=score(10:12,1:2);
x4=score4(:,1); y4=score4(:,2);
score5=score(13:15,1:2);
x5=score5(:,1); y5=score5(:,2);

%find centers for data from each cluster
meanx1 = mean(x1);
meany1 = mean(y1);
meanx2 = mean(x2);
meany2 = mean(y2);
meanx3 = mean(x3);
meany3 = mean(y3);
meanx4 = mean(x4);
meany4 = mean(y4);
meanx5 = mean(x5);
meany5 = mean(y5);

%confidence intervals

% x is a vector, matrix, or any numeric array of data. NaNs are ignored.
% p is the confidence level (ie, 95 for 95% CI)
% The output is 1x2 vector showing the [lower,upper] interval values.

p=95; 
%find upper and lower bounds for 95% confidence in x and y directions
% CIFcn_x1 = @(x1,p)prctile(x1,abs([0,100]-(100-p)/2));
CIFcn_x1 = prctile(x1,abs([0,100]-(100-p)/2));
CIFcn_y1 = prctile(y1,abs([0,100]-(100-p)/2));
CIFcn_x2 = prctile(x2,abs([0,100]-(100-p)/2));
CIFcn_y2 = prctile(y2,abs([0,100]-(100-p)/2));
CIFcn_x3 = prctile(x3,abs([0,100]-(100-p)/2));
CIFcn_y3 = prctile(y3,abs([0,100]-(100-p)/2));
CIFcn_x4 = prctile(x4,abs([0,100]-(100-p)/2));
CIFcn_y4 = prctile(y4,abs([0,100]-(100-p)/2));
CIFcn_x5 = prctile(x5,abs([0,100]-(100-p)/2));
CIFcn_y5 = prctile(y5,abs([0,100]-(100-p)/2));

CI_x1 = abs((CIFcn_x1(1)-CIFcn_x1(2))/2); 
CI_y1 = abs((CIFcn_y1(1)-CIFcn_y1(2))/2); 
CI_x2 = abs((CIFcn_x2(1)-CIFcn_x2(2))/2); 
CI_y2 = abs((CIFcn_y2(1)-CIFcn_y2(2))/2); 
CI_x3 = abs((CIFcn_x3(1)-CIFcn_x3(2))/2); 
CI_y3 = abs((CIFcn_y3(1)-CIFcn_y3(2))/2); 
CI_x4 = abs((CIFcn_x4(1)-CIFcn_x4(2))/2); 
CI_y4 = abs((CIFcn_y4(1)-CIFcn_y4(2))/2); 
CI_x5 = abs((CIFcn_x5(1)-CIFcn_x5(2))/2); 
CI_y5 = abs((CIFcn_y5(1)-CIFcn_y5(2))/2); 

mean1 = [meanx1,meany1];
std1 = [CI_x1,CI_y1];
mean2 = [meanx2,meany2];
std2 = [CI_x2,CI_y2];
mean3 = [meanx3,meany3];
std3 = [CI_x3,CI_y3];
mean4 = [meanx4,meany4];
std4 = [CI_x4,CI_y4];
mean5 = [meanx5,meany5];
std5 = [CI_x5,CI_y5];

%add ellipses 
E = plotEllipses(mean1,std1); 
E.FaceColor = [1 0 0 .2]; 
E.EdgeColor = [1 0 0]; 
E.LineWidth = 0.5; 
 
E = plotEllipses(mean2,std2); 
E.FaceColor = [0 0.502 0 .2]; 
E.EdgeColor = [0 0.502 0]; 
E.LineWidth = 0.5; 

E = plotEllipses(mean3,std3); 
E.FaceColor = [0.502 0 0.502 .1]; 
E.EdgeColor = [0.502 0 0.502]; 
E.LineWidth = 0.5; 

E = plotEllipses(mean4,std4); 
E.FaceColor = [1 0.502 0 .2]; 
E.EdgeColor =[1 0.502 0]; 
E.LineWidth = 0.5; 
 
E = plotEllipses(mean5,std5); 
E.FaceColor = [0.0275 0.4941 0.5922 .2]; 
E.EdgeColor = [0.0275 0.4941 0.5922]; 
E.LineWidth = 0.5; 


%Create 3-D score plot 
figure('Name','3D Score Plot + bi-plot')
C = [1 0 0; 1 0 0; 1 0 0; 0 0.502 0; 0 0.502 0; 0 0.502 0; 0.502 0 0.502; 0.502 0 0.502; 0.502 0 0.502; 1 0.502 0; 1 0.502 0; 1 0.502 0; 0.0275 0.4941 0.5922; 0.0275 0.4941 0.5922; 0.0275 0.4941 0.5922];

h= scatter3(score(:,1),score(:,2),score(:,3),'filled')
h.CData = C; 

% Create loading plot and add to score plot (i.e. biplot)
% Note: biplot allows us to visualize both the orthonormal principal component coefficients for each variable (i.e. timepoint) and the principal component scores (first 2) for each observation in a single plot.
hold on
substrates = {'Collagenase Peptide', 'Gelatinase Peptide', 'P3 NPro', 'P2 NAla', 'P1 NAsn', 'P3p NAla', 'P3 NAla', 'P1 NAla', 'P1 NAsn P3p NAla', 'P3 NAla P1 NAsn', 'P3 NAla P1 NAla'};
p = biplot(coeff(:,1:3),'VarLabels', substrates);

xlabel('Component 1 (82.6%)')
ylabel('Component 2 (12.1%)')
zlabel('Component 3 (3.0%)')



%% Reconstruct data and Train a kNN using the first idx components.

score_recon= score*coeff' +  repmat(mu,15,1); %reconstruct data using PCs
scoreTrain95 = score_recon(:,1:idx);


rng(1); %For reproducibility
cv = cvpartition(YTrain, 'KFold',3,'Stratify',true)
n_neighbors=2;
mdl = fitcknn(scoreTrain95,YTrain,'NumNeighbors',n_neighbors,'Distance','euclidean','CVPartition',cv); %mdl is a kNN classification model

classError=kfoldLoss(mdl) %examine model's classification error

[Predict_Labels,Predict_Scores]=kfoldPredict(mdl); %This evaluates how well the model does on each validation dataset during the cross-validation

figure ('Name', 'Confusion Matrix- kNN Model');
cm=confusionchart(YTrain, Predict_Labels,'RowSummary','row-normalized','ColumnSummary','column-normalized');%Confusion matrix of model


