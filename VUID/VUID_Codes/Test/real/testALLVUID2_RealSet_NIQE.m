% The following code is to test the proposed VUID(Deep Variational
% Ultrasound Image Despeckling)-TwoInput with the additional SigmaMap
% Note:VUID is composed of a Despeckling network, and a SigmaSq network.
% both will be implemented individually.
% Ref:[1] Z. Yue, H. Yong, Q. Zhao, D. Meng, and L. Zhang, "Variational denoising network: Toward blind noise modeling and removal,"
%        in Proceedings of Advances in Neural Information Processing Systems (NeurIPS), 2019, pp. 1688–1699.
%     [2] Yue, Zongsheng, et al. "Deep Variational Network Toward Blind Image Restoration." arXiv preprint arXiv:2008.10796 (2020).
% author:Cui wenchao; Date:2023.09.09

%%%%%%%%%%%%%%%%%%%%%%   Prepare test dataset   %%%%%%%%%%%%%%%%%%%%%%%%%%
% noisyDirName='D:\Continued from 2019\US despeckling experiment\2023 VUID\RealTestSetBUID\noisyImages';
% VUIDnetOutDirName='D:\Continued from 2019\US despeckling experiment\2023 VUID\RealTestSetBUID\VUIDnetOutImages';

noisyDirName='D:\Continued from 2019\US despeckling experiment\2023 VUID\RealTestSetBUSI\noisyImages';
VUIDnetOutDirName='D:\Continued from 2019\US despeckling experiment\2023 VUID\RealTestSetBUSI\VUIDnetOutImages';

imdsNoisy = imageDatastore(noisyDirName,FileExtensions=".mat",ReadFcn=@matRead);
imdsVUIDnetOut = imageDatastore(VUIDnetOutDirName,FileExtensions=".mat",ReadFcn=@matRead);

%%%%%%%%%%%%%%%%%%%%%%%%   Load the pretrained VUID  %%%%%%%%%%%%%%%%%%%%%%


DespecklerDirName='D:\Continued from 2019\US despeckling experiment\2023 VUID\VUID_Unet_DnCNN5_TwoInput_sumloss_RecLoss_ResLearning_Var3_Patch32_BUSI_0327\DenoiserNet';
imdsDespeckler=imageDatastore(DespecklerDirName,FileExtensions=".mat",ReadFcn=@matRead);
SigmaSqDirName='D:\Continued from 2019\US despeckling experiment\2023 VUID\VUID_Unet_DnCNN5_TwoInput_sumloss_RecLoss_ResLearning_Var3_Patch32_BUSI_0327\SigmaSqNet';
imdsSigmaSq=imageDatastore(SigmaSqDirName,FileExtensions=".mat",ReadFcn=@matRead);


%%%%%%%%%%%%%%%%%%%%%%%%   Predict the output of the VUID   %%%%%%%%%%%%%%%
while hasdata(imdsDespeckler)

    [DespeckleNet,info1] = read(imdsDespeckler);
    [~,fileNameDenoi,~] = fileparts(info1.Filename);
    [SigmaSqNet,info2] = read(imdsSigmaSq);
    [~,fileNameEncod,~] = fileparts(info2.Filename);

    numImages=numel(imdsNoisy.Files);
    cleanPredictions=cell(numImages);
    cleanTruth=cell(numImages);
    NoisyImages=cell(numImages);
    niqeNoise=[];
    piqeNoise=[];
    brisqueNoise=[];

    niqeTruth=[];
    piqeTruth=[];
    brisqueTruth=[];

    niqeDespeckle=[];
    piqeDespeckle=[];
    brisqueDespeckle=[];

    for i=1:numImages
        Noisy=matRead(imdsNoisy.Files{i});
        NoisyImages{i}=Noisy;


        MiuSigmaSq=matRead(imdsVUIDnetOut.Files{i});
        Miu=MiuSigmaSq(:,:,1:3);
        cleanTru=Miu;
        cleanTruth{i}=cleanTru;

        fun=@(block_struct) predVUID(block_struct.data,DespeckleNet,SigmaSqNet);
        cleanImg=blockproc(Noisy,[108 108],fun,"BorderSize",[10 10],"PadMethod","symmetric",'PadPartialBlocks',true);
        cleanImg=double(cleanImg);
        cleanImg=imcrop(cleanImg,[0 0 size(Noisy,2) size(Noisy,1)]);
        cleanImg=Noisy+cleanImg;%Residual Learning
        cleanImg=max(min(cleanImg,1),0);
        cleanPredictions{i}=cleanImg;
        
        niqeDenoi=niqe(cleanImg);
        niqeDespeckle=cat(2,niqeDespeckle,niqeDenoi);
        piqeDenoi=piqe(cleanImg);
        piqeDespeckle=cat(2,piqeDespeckle,piqeDenoi);
        brisqueDenoi=brisque(cleanImg);
        brisqueDespeckle=cat(2,brisqueDespeckle,brisqueDenoi);

        niqeNoi=niqe(Noisy);
        niqeNoise=cat(2,niqeNoise,niqeNoi);
        piqeNoi=piqe(Noisy);
        piqeNoise=cat(2,piqeNoise,piqeNoi);
        brisqueNoi=brisque(Noisy);
        brisqueNoise=cat(2,brisqueNoise,brisqueNoi);
        

        niqeTru=niqe(cleanTru);
        niqeTruth=cat(2,niqeTruth,niqeTru);
        piqeTru=piqe(cleanTru);
        piqeTruth=cat(2,piqeTruth,piqeTru);
        brisqueTru=brisque(cleanTru);
        brisqueTruth=cat(2,brisqueTruth,brisqueTru);

    end

    niqeAvgDespeckle=mean(niqeDespeckle(:));
    piqeAvgDespeckle=mean(piqeDespeckle(:));
    brisqueAvgDespeckle=mean(brisqueDespeckle(:));
    niqeStdDespeckle=std(niqeDespeckle(:));
    piqeStdDespeckle=std(piqeDespeckle(:));
    brisqueStdDespeckle=std(brisqueDespeckle(:));

    niqeAvgNoise=mean(niqeNoise(:));
    piqeAvgNoise=mean(piqeNoise(:));
    brisqueAvgNoise=mean(brisqueNoise(:));
    niqeStdNoise=std(niqeNoise(:));
    piqeStdNoise=std(piqeNoise(:));
    brisqueStdNoise=std(brisqueNoise(:));

    niqeAvgTruth=mean(niqeTruth(:));
    piqeAvgTruth=mean(piqeTruth(:));
    brisqueAvgTruth=mean(brisqueTruth(:));
    niqeStdTruth=std(niqeTruth(:));
    piqeStdTruth=std(piqeTruth(:));
    brisqueStdTruth=std(brisqueTruth(:));

    disp('-----------------------------------------------------------');
    str1=sprintf('Average Noisy NIQE: %.2f±%.2f  PIQE: %.2f±%.2f  BRISQUE: %.2f±%.2f',...
        niqeAvgNoise,niqeStdNoise,piqeAvgNoise,piqeStdNoise,brisqueAvgNoise,brisqueStdNoise);
    disp(str1);

    str2=sprintf('Average Truth NIQE: %.2f±%.2f  PIQE: %.2f±%.2f  BRISQUE: %.2f±%.2f',...
        niqeAvgTruth,niqeStdTruth,piqeAvgTruth,piqeStdTruth,brisqueAvgTruth,brisqueStdTruth);
    disp(str2);

    disp(fileNameDenoi);
    
    str3=sprintf('Average Despeckle NIQE: %.2f±%.2f  PIQE: %.2f±%.2f  BRISQUE: %.2f±%.2f',...
        niqeAvgDespeckle,niqeStdDespeckle,piqeAvgDespeckle,piqeStdDespeckle,brisqueAvgDespeckle,brisqueStdDespeckle);
    disp(str3);


    %%% Display denoising results with PSNR and SSIM

    % idx = randperm(numImages,8);
    % NoisyI=[];
    % denoiI=[];
    % truthI=[];
    % for i=1:8
    %     NoisyI=cat(4,NoisyI,NoisyImages(:,:,:,idx(i)));
    %     denoiI=cat(4,denoiI,cleanPredictions(:,:,:,idx(i)));
    %     truthI=cat(4,truthI,cleanTruth(:,:,:,idx(i)));
    % end
    % 
    % figure
    % subplot(3,1,1)
    % Inoisy=imtile(NoisyI,GridSize=[1 8],ThumbnailSize=[]);
    % strNoisy=sprintf('PSNR = %.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f \n SSIM = %.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f ',...
    %     psnrNoise(idx(1)),psnrNoise(idx(2)),psnrNoise(idx(3)),psnrNoise(idx(4)),psnrNoise(idx(5)),...
    %     psnrNoise(idx(6)),psnrNoise(idx(7)),psnrNoise(idx(8)),ssimNoise(idx(1)),ssimNoise(idx(2)),...
    %     ssimNoise(idx(3)),ssimNoise(idx(4)),ssimNoise(idx(5)),ssimNoise(idx(6)),ssimNoise(idx(7)),...
    %     ssimNoise(idx(8)));
    % imshow(Inoisy);title(['Noisy Images ' strNoisy]);
    % subplot(3,1,2)
    % Idenoise=imtile(denoiI,GridSize=[1 8],ThumbnailSize=[]);
    % strPSNRSSIM=sprintf('PSNR = %.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f \n SSIM = %.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f ',...
    %     psnrDenoise(idx(1)),psnrDenoise(idx(2)),psnrDenoise(idx(3)),psnrDenoise(idx(4)),psnrDenoise(idx(5)),...
    %     psnrDenoise(idx(6)),psnrDenoise(idx(7)),psnrDenoise(idx(8)),ssimDenoise(idx(1)),ssimDenoise(idx(2)),...
    %     ssimDenoise(idx(3)),ssimDenoise(idx(4)),ssimDenoise(idx(5)),ssimDenoise(idx(6)),ssimDenoise(idx(7)),...
    %     ssimDenoise(idx(8)));
    % 
    % imshow(Idenoise);title(['Denoised Images ' strPSNRSSIM]);
    % subplot(3,1,3)
    % Iclean=imtile(truthI,GridSize=[1 8],ThumbnailSize=[]);
    % imshow(Iclean);title('Truth Images');
end

function CleanPre=predVUID(Img2D,DespeckleNet,SigmaSqnet)
%function CleanPre=predVUID(Img2D,DespeckleNet)
Img3D=reshape(Img2D,128, 128, []);
inputImg=dlarray(Img3D,'SSC');
if canUseGPU
    inputImg=gpuArray(inputImg);
end

p=7;
alpha0=0.5*p*p;
LogBeta=predict(SigmaSqnet,inputImg);
Beta=exp(LogBeta);

CleanI = predict(DespeckleNet,inputImg,Beta./alpha0);
CleanI=extractdata(gather(CleanI));
CleanPre=squeeze(CleanI);
end






