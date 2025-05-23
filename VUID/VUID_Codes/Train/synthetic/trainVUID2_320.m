% The following code is to implement and train the the proposed VUID(Deep
% Variational Ultrasound Image Despeckling)-TwoInput with SigmaMap
% Note:VUID is composed of a Despeckling network, and a SigmaSq network.
% both will be implemented individually.
% Ref:[1] Z. Yue, H. Yong, Q. Zhao, D. Meng, and L. Zhang, "Variational denoising network: Toward blind noise modeling and removal,"
%        in Proceedings of Advances in Neural Information Processing Systems (NeurIPS), 2019, pp. 1688–1699.
%     [2] Yue, Zongsheng, et al. "Deep Variational Network Toward Blind Image Restoration." arXiv preprint arXiv:2008.10796 (2020).
% author:Cui wenchao; Date:2023.09.09

%%%%%%%%%%%%   Prepare training dataset   %%%%%%%%%%%%
noisyDirName='D:\Continued from 2019\US despeckling experiment\2019 VDN\SynTrainingSet\noisyImages';
VUIDnetOutDirName='D:\Continued from 2019\US despeckling experiment\2019 VDN\SynTrainingSet\VUIDnetOutImages';

imdsNoisy = imageDatastore(noisyDirName,FileExtensions=".mat",ReadFcn=@matRead);
imdsVUIDnetOut = imageDatastore(VUIDnetOutDirName,FileExtensions=".mat",ReadFcn=@matRead);

augmenter = imageDataAugmenter(RandXReflection=true);

patchSize = 128;
patchesPerImage = 16;
dsTrain = randomPatchExtractionDatastore(imdsNoisy,imdsVUIDnetOut,patchSize, ...
    PatchesPerImage=patchesPerImage, ...
    DataAugmentation=augmenter);
dsTrain.MiniBatchSize = 32;

%%%%%%    View the training samples   %%%%%%%%%%%%%%%%
dsTrain=shuffle(dsTrain);
inputBatch = preview(dsTrain);
noisyimages=inputBatch.InputImage;
cleanimages=noisyimages;
for i=1:8
    cleanimages{i}=inputBatch.ResponseImage{i}(:,:,1:3);
end
minibatch = cat(2,noisyimages,cleanimages);
figure
montage(minibatch,Size=[2 8])
title("Noisy (Up) and Clean (Down)")
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%   Prepare validation dataset   %%%%%%%%%%%%
% noisyDirName='D:\Continued from 2019\US despeckling experiment\2023 VUID\SynValSet\noisyImages';
% VUIDnetOutDirName='D:\Continued from 2019\US despeckling experiment\2023 VUID\SynValSet\VUIDnetOutImages';
% 
% imdsNoisy = imageDatastore(noisyDirName,FileExtensions=".mat",ReadFcn=@matRead);
% imdsVUIDnetOut = imageDatastore(VUIDnetOutDirName,FileExtensions=".mat",ReadFcn=@matRead);
% 
% dsVal = randomPatchExtractionDatastore(imdsNoisy,imdsVUIDnetOut,patchSize, ...
%     PatchesPerImage=patchesPerImage);
% dsVal.MiniBatchSize = 32;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%% Training from Scratch %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%  DespecklingNet based on ResUnet %%%%%%%%%%%%%%%%%%
 DespecklingNet=load("VDN_Unet_TwoInput.mat");


lgraphDespecklingNet=DespecklingNet.lgraph_1;
DespecklingNet=dlnetwork(lgraphDespecklingNet);
%analyzeNetwork(DespecklingNet);% Check the DespecklingNet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%       SigmaSqNet based on DnCNN      %%%%%%%%%%%%%%%%%
SigmaSqNet=load("SigmaSqNet.mat");
layersSigmaSqNet=SigmaSqNet.layers_1;
lgraphSigmaSqNet=layerGraph(layersSigmaSqNet);



SigmaSqNet=dlnetwork(lgraphSigmaSqNet);
%analyzeNetwork(SigmaSqNet);% Check the SigmaSqNet
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%  custom training loops   %%%%%%%%%%%%%%%%%
numEpochs = 30;
miniBatchSize = 20;
learnRate = 0.0001;
learnRateDropPeriod = 10;
learnRateDropFactor = 0.5;
gradientDecayFactor = 0.90;
squaredGradientDecayFactor = 0.999;
epsilon=1e-8;
validationFrequency = 200;
chechpointFrequency = 1;
verbose = true;
verboseFrequency = 100;


mbq = minibatchqueue(dsTrain, ...
    MiniBatchSize=miniBatchSize, ...
    MiniBatchFcn=@preprocessMiniBatch, ...
    MiniBatchFormat=["SSCB" "SSCB"]);

% mbqVal = minibatchqueue(dsVal, ...
%     MiniBatchSize=miniBatchSize,...
%     MiniBatchFcn=@preprocessMiniBatch, ...
%     MiniBatchFormat=["SSCB" "SSCB"]);

trailingAvgDespecklingNet = [];
trailingAvgSqDespecklingNet = [];
trailingAvgSigmaSqNet = [];
trailingAvgSqSigmaSqNet = [];

numObservationsTrain = dsTrain.NumObservations;
numIterationsPerEpoch = ceil(numObservationsTrain/miniBatchSize);
numIterations = numEpochs*numIterationsPerEpoch;

% monitor = trainingProgressMonitor( ...
%     Metrics=["TrainingLoss","ValidationLoss","TrainingPSNR","ValidationPSNR"]);
% groupSubPlot(monitor,"Loss",["TrainingLoss","ValidationLoss"]);
% groupSubPlot(monitor,"PSNR",["TrainingPSNR","ValidationPSNR"]);

monitor = trainingProgressMonitor( ...
    Metrics=["TrainingLoss","TrainingPSNR"]);
groupSubPlot(monitor,"Loss","TrainingLoss");
groupSubPlot(monitor,"PSNR","TrainingPSNR");

monitor.Info = ["LearningRate","Epoch","Iteration","ExecutionEnvironment"];
monitor.XLabel = "Iteration";
monitor.Status = "Configuring";
monitor.Progress = 0;
executionEnvironment = "auto";

if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    updateInfo(monitor,ExecutionEnvironment="GPU");
else
    updateInfo(monitor,ExecutionEnvironment="CPU");
end

%%%%%%%%%%%%    training loops     %%%%%%%%%%%%%
epoch = 0;
iteration = 0;

monitor.Status = "Running";

disp("|========================================================================================|")
disp("|  Epoch  |  Iteration  |  Time Elapsed  |  Training    |  Mini-batch  |  Base Learning  |")
disp("|         |             |   (hh:mm:ss)   |    PSNR      |     Loss     |      Rate       |")
disp("|========================================================================================|")

start = tic;

% Loop over epochs.
while epoch < numEpochs && ~monitor.Stop
    epoch = epoch + 1;

    % Shuffle data.
    shuffle(mbq);

    % Loop over mini-batches.
    while hasdata(mbq) && ~monitor.Stop
        iteration = iteration + 1;

        % Read mini-batch of data.
        [X,T] = next(mbq);

        % Evaluate the gradients of the loss with respect to the learnable
        % parameters, the generator state, and the network scores using
        % dlfeval and the modelLoss function.
        [lossVUID,gradientsDespecklingNet,gradientsSigmaSqNet,stateDespecklingNet,stateSigmaSqNet,...
            TrainingPSNR] = dlfeval(@modelLoss,DespecklingNet,SigmaSqNet,X,T);
        DespecklingNet.State=stateDespecklingNet;
        SigmaSqNet.State=stateSigmaSqNet;

        % Update the SigmaSqNet network parameters.
        [SigmaSqNet,trailingAvgSigmaSqNet,trailingAvgSqSigmaSqNet] = adamupdate(SigmaSqNet, gradientsSigmaSqNet, ...
            trailingAvgSigmaSqNet, trailingAvgSqSigmaSqNet, iteration, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);

        % Update the DespecklingNet network parameters.
        [DespecklingNet,trailingAvgDespecklingNet,trailingAvgSqDespecklingNet] = adamupdate(DespecklingNet, gradientsDespecklingNet, ...
            trailingAvgDespecklingNet, trailingAvgSqDespecklingNet, iteration, ...
            learnRate, gradientDecayFactor, squaredGradientDecayFactor);

        % Record training loss and PSNR.
        recordMetrics(monitor,iteration, TrainingLoss=lossVUID, TrainingPSNR=TrainingPSNR);

        % Update learning rate, epoch, and iteration information values.
        updateInfo(monitor, ...
            LearningRate=learnRate, ...
            Epoch=string(epoch) + " of " + string(numEpochs), ...
            Iteration=string(iteration) + " of " + string(numIterations));

         if verbose && (iteration == 1 || mod(iteration,verboseFrequency) == 0)
            D = duration(0,0,toc(start),'Format','hh:mm:ss');

            
            disp("| " + ...
                pad(num2str(epoch),7,'both') + " | " + ...
                pad(num2str(iteration),11,'both') + " | " + ...
                pad(string(D),14,'both') + " | " + ...
                pad(num2str(TrainingPSNR),12,'both') + " | " + ...
                pad(num2str(lossVUID),12,'both') + " | " + ...
                pad(num2str(learnRate),15,'both') + " |")
        end

        % % Record validation loss and accuracy.
        % if iteration == 1 || mod(iteration,validationFrequency)==0
        %     [ValidationLoss,ValidationPSNR] = modelPredictions(DespecklingNet,SigmaSqNet,mbqVal);
        %     ValidationLoss=mean(ValidationLoss(:));
        %     ValidationPSNR=mean(ValidationPSNR(:));
        % 
        %     recordMetrics(monitor,iteration, ValidationLoss=ValidationLoss, ValidationPSNR=ValidationPSNR);
        % end

        % Update progress percentage.
        monitor.Progress = 100*iteration/numIterations;
    end

    % Determine learning rate for epoch-based piecewise linear  schedule.
    if mod(epoch,learnRateDropPeriod) == 0
        learnRate = learnRate * learnRateDropFactor;
    end

    % Save checkpoint networks
    if mod(epoch,chechpointFrequency)==0
        path="D:\Continued from 2019\US despeckling experiment\2023 VUID";
        checkpointPath = fullfile(path,"checkpoints");
        if ~exist(checkpointPath,"dir")
            mkdir(checkpointPath);
        end
        if ~isempty(checkpointPath)
            D = string(datetime("now",Format="yyyy_MM_dd__HH_mm_ss"));
            filenameD = "DespecklingNet_checkpoint_Epoch" + epoch + "_" + D + ".mat";
            save(fullfile(checkpointPath,filenameD),"DespecklingNet");
            filenameSigma = "SigmaSqNet_checkpoint_Epoch" + epoch + "_" + D + ".mat";
            save(fullfile(checkpointPath,filenameSigma),"SigmaSqNet");
        end

    end

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if monitor.Stop == 1
    monitor.Status = "Training stopped";
else
    monitor.Status = "Training complete";
end
%%%%%%%%%%%%%%      Save the trained Dnet and Snet     %%%%%%%%%%%%%%%
modelDateTime = string(datetime("now",Format="yyyy-MM-dd-HH-mm-ss"));
save("trainedDespecklingNet-"+"Epoch"+num2str(epoch)+"-"+modelDateTime+".mat","DespecklingNet");
save("trainedSigmaSqNet-"+"Epoch"+num2str(epoch)+"-"+modelDateTime+".mat","SigmaSqNet");


%%%%%%%%%%%%  Supporting Functions  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [X,T] = preprocessMiniBatch(dataX,dataT)
% Concatenate mini-batch
X = cat(4,dataX{:});
T = cat(4,dataT{:});
end

function gradients = thresholdL2Norm(gradients,gradientThreshold)

gradientNorm = sqrt(sum(gradients(:).^2));
if gradientNorm > gradientThreshold
    gradients = gradients * (gradientThreshold / gradientNorm);
end

end



function [lossVUID,gradientsDespecklingNet,gradientsSigmaSqNet,stateDespecklingNet,stateSigmaSqNet,TrainingPSNR] = ...
    modelLoss(DespecklingNet,SigmaSqNet,X,T)

% epsilon1=2.236e-4;%variance=1.0e-7;
 epsilon1=0.3536e-3;%variance=0.25e-6;
% epsilon1=0.5e-3;%variance=0.5e-6;
% epsilon1=7.07e-4;%variance=1.0e-6;
% epsilon1=2.236e-3;%variance=1.0e-5;
% epsilon1=7.07e-3;%variance=1.0e-4;
p=7;
alpha0=0.5*p*p;
% Calculate the predictions for the DespecklingNet, and SigmaSqNet.
[LogBeta,stateSigmaSqNet] = forward(SigmaSqNet,X);

logmin=log(1e-8);logmax=log(10000);
LogBeta=max(min(LogBeta,logmax),logmin);
Beta=exp(LogBeta);


% Calculate the predictions for the DespecklingNet, and SigmaSqNet.
[Miu,stateDespecklingNet] = forward(DespecklingNet,X,Beta./alpha0);
% Miu=max(min(Miu,1),0);

Miu=X+Miu;

%Reparameteration to reconstruct the Noisy image
 X_Rec=Miu+sqrt(Beta./alpha0).*randn(size(Miu));


% Calculate the loss including likelihood loss (LH_loss), KL_Miu loss (KL_Miu), and KL_SigmaSq loss.
LH_loss=-0.5*log(2*pi)-0.5.*log(Beta)+0.5*3.1356-0.5*23.5./Beta.*((X-Miu).^2+2*epsilon1*epsilon1);%psi(0.5P^2-1)=psi(23.5)=3.1356;p=7;

KL_Miu=exp(-abs(Miu-T(:,:,1:3,:))./epsilon1)+abs(Miu-T(:,:,1:3,:))./epsilon1-1;

KL_SigmaSq=(alpha0-1).*(log(Beta+eps)-log(alpha0.*T(:,:,4:6,:)+eps))+(alpha0-1).*(alpha0.*T(:,:,4:6,:)./Beta)-(alpha0-1);

Rec_loss=abs(X_Rec-X);

% lossVUID=mean(KL_SigmaSq,"all")+mean(KL_Miu,"all")-mean(LH_loss,"all");
% lossVUID=sum(KL_SigmaSq,"all")+sum(KL_Miu,"all")-sum(LH_loss,"all");
 lossVUID=sum(Rec_loss,"all")+sum(KL_SigmaSq,"all")+sum(KL_Miu,"all")-sum(LH_loss,"all");

TrainingPSNR=-10*log((T(:,:,1:3,:)-Miu).^2+eps);
TrainingPSNR=mean(TrainingPSNR,"all");

% Calculate the gradients with respect to the loss.
[gradientsSigmaSqNet,gradientsDespecklingNet] = dlgradient(lossVUID,...
    SigmaSqNet.Learnables,DespecklingNet.Learnables);
gradientsSigmaSqNet = dlupdate(@(g) thresholdL2Norm(g, 1e4),gradientsSigmaSqNet);
gradientsDespecklingNet = dlupdate(@(g) thresholdL2Norm(g, 1e4),gradientsDespecklingNet);


end





