%The following codes are to compute the NIQE/PIQE indices of BUID and BUSI despeckled by several despeckling methods

%clc;clear;

DespeckledUSDir='D:\Continued from 2019\US despeckling experiment\BUSI_Seg\test\USimages';
imdsDeUS=imageDatastore(DespeckledUSDir);

NIQEall=[];
PIQEall=[];
ENLall=[];
AGMall=[];

while hasdata(imdsDeUS)
    [DeUSimage,info]=read(imdsDeUS);
    NIQE=niqe(DeUSimage);
    PIQE=piqe(DeUSimage);
    [ENL,AGM]=compute_ENL_AGM(DeUSimage);
    NIQEall=cat(2,NIQEall,NIQE);
    PIQEall=cat(2,PIQEall,PIQE);
    ENLall=cat(2,ENLall,ENL);
    AGMall=cat(2,AGMall,AGM);
end

NIQEavg=mean(NIQEall);
NIQEstd=std(NIQEall);
PIQEavg=mean(PIQEall);
PIQEstd=std(PIQEall);
ENLavg=mean(ENLall);
ENLstd=std(ENLall);
AGMavg=mean(AGMall);
AGMstd=std(AGMall);
disp('-----------------------------------------------------------');
str1=sprintf('Average NIQE: %.4f±%.4f PIQE: %.4f±%.4f \nENL: %.4f±%.4f AGM: %.4f±%.4f',...
    NIQEavg,NIQEstd,PIQEavg,PIQEstd,ENLavg,ENLstd,AGMavg,AGMstd);
disp(str1);


function [ENL,AGM]=compute_ENL_AGM(Im_Despeckled)
Im_Despeckled=im2double(Im_Despeckled);
Im_Despeckled=rgb2gray(Im_Despeckled);
Im_Despeckled=rescale(Im_Despeckled,0,255);
E_g=mean2(Im_Despeckled);
std_g=std2(Im_Despeckled);
ENL=E_g^2/(std_g^2);
[Gmag,~] = imgradient(Im_Despeckled);
AGM=mean2(Gmag);
end