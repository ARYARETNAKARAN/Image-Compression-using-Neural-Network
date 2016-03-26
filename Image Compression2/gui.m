function varargout = unanggui(varargin)
% UNANGGUI MATLAB code for unanggui.fig
%      UNANGGUI, by itself, creates a new UNANGGUI or raises the existing
%      singleton*.
%
%      H = UNANGGUI returns the handle to a new UNANGGUI or the handle to
%      the existing singleton*.
%
%      UNANGGUI('CALLBACK',hObject,eventData,handles,...) calls the local
%      function named CALLBACK in UNANGGUI.M with the given input arguments.
%
%      UNANGGUI('Property','Value',...) creates a new UNANGGUI or raises the
%      existing singleton*.  Starting from the left, property value pairs are
%      applied to the GUI before unanggui_OpeningFcn gets called.  An
%      unrecognized property name or invalid value makes property application
%      stop.  All inputs are passed to unanggui_OpeningFcn via varargin.
%
%      *See GUI Options on GUIDE's Tools menu.  Choose "GUI allows only one
%      instance to run (singleton)".
%
% See also: GUIDE, GUIDATA, GUIHANDLES

% Edit the above text to modify the response to help unanggui

% Last Modified by GUIDE v2.5 16-Sep-2014 15:10:11

% Begin initialization code - DO NOT EDIT
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @unanggui_OpeningFcn, ...
                   'gui_OutputFcn',  @unanggui_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT


% --- Executes just before unanggui is made visible.
function unanggui_OpeningFcn(hObject, eventdata, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to unanggui (see VARARGIN)

% Choose default command line output for unanggui
handles.output = hObject;



% Update handles structure
guidata(hObject, handles);

% UIWAIT makes unanggui wait for user response (see UIRESUME)
% uiwait(handles.figure1);


% --- Outputs from this function are returned to the command line.
function varargout = unanggui_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
varargout{1} = handles.output;



% --- Executes on button press in browse.
function browse_Callback(hObject, eventdata, handles)
% hObject    handle to browse (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
%[Filename] = uigetfile({'*.bmp';'*.jpg';'*.png';'*.*'},'Select Picture');
%axes(handles.showimage);
%imshow(Filename);

[Filename, Pathname] = uigetfile({'*.bmp';'*.jpg';'*.png';'*.*'},'Select Picture');
%fullpathname = strcat(Pathname, Filename);
%text = fileread(fullpathname);
axes(handles.showimage);
imshow(Filename);


% --- Executes on button press in compress.
function compress_Callback(hObject, eventdata, handles)
% hObject    handle to compress (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

%clc;
%clear all;
%close all;
%function Codec

    set(handles.compresstext,'String','Start Compressing/Decompressing . . . ');
    [In,I,H]=ReadParams;
    
    %I=64;                      % Number of nerons in Input and output layer
    %H=16;                      % Number of nerons in hidden layer
    Sqrt_H=fix(sqrt(H));        % Input and Output Block size:  Sqrt_H* Sqrt_H        
    Sqrt_I=fix(sqrt(I));        % Compressed Block size:  Sqrt_I* Sqrt_I

    Time = clock;        % Start execution clock
    
   % showimage = calllib(browse);
   % axes(handles.showimage);
   % Input = imread(showimage);
    %Filename = functions(@browse);
    %Filename = get(handles.showimage);
    %getaxes(handles.showimage);
    %In = get(gcf,'CurrentAxes');
    %In=imread(handles.showimage);
    [M,N]=size(getimage(handles.showimage));
    No=M*N/I;                           % Number of input blocks
    
    [v,w,v_b,w_b]=ReadWeights(I,H);     % Return weights of network 64_16_64

    mrg=zeros(M/2,N/2);                 % Compressed image  
    Dmrg=zeros(M,N);                    % Decompressed image  
    
    g = 'Complete';
    set(handles.compresstext,'String',g);
    
    for u=1:No
        [x]=NextPattern(getimage(handles.showimage),I,u);     % Read next block of image(x vector) that the block size=64
        x=double(x)/256;                % Normalize to [0,1]
        
        h_in=(x'*v)'+v_b;               % Input of hidden layer
        h=f1(h_in);                     % Output of hidden layer
        % h is the compressed data
        
        k=1;
        for i=1:  Sqrt_H
            for j=1: Sqrt_H
                tmp(i,j)=h(k);
                k=k+1;
            end
        end
                
        mrg=AddPattern(mrg,tmp,M/2,N/2,u);
        
        y_in=w_b+(h'*w)';                  % Input of output layer
        y=f1(y_in);                     % Output of output layer
        % y is decompressed data
        k=1;
        for i=1: Sqrt_I
            for j=1: Sqrt_I
                tmp(i,j)=y(k);
                k=k+1;
            end
        end
        Dmrg=AddPattern(Dmrg,tmp,M,N,u);
        clear tmp;
    end
        
    Dmrg=Dmrg*256;
    mrg=mrg*256;
    
    % *************************************************
    % ****             Outputs                      ***
    % *************************************************
   % Compute the PSNR
    Psnr=PSNR(getimage(handles.showimage),Dmrg);
    snr=SNR(getimage(handles.showimage),Dmrg);
    nmse=NMSE(getimage(handles.showimage),Dmrg);

    
    % Computing Bit rate
%    BitRate(BlockSize,NoOfBlocks,NoOfHiddenNeroun,NoOfBitsOut,NoOfBitsWeight)
    NoOfBitsOut=8;
    NoOfBitsWeight=8;
    Bitrate=BitRate(I,No,H,NoOfBitsOut,NoOfBitsWeight);
    
    
    a = ['PSNR= ',num2str(Psnr),'dB' ];
    set(handles.psnr,'String',a);
    b =['SNR= ',num2str(snr),' dB'];
    set(handles.snr,'String',b);
    c =['NMSE= ',num2str(nmse)];
    set(handles.nmse,'String',c);
    d=['Bit rate(all)= ',num2str(Bitrate),' bpp'];
    set(handles.bitrateall,'String',d);
    e =['Bit rate(CR)= ',num2str(H/I),' bpp'];
    set(handles.bitratecr,'String',e);
    
     
    axes(handles.originalimage)
    imshow(getimage(handles.showimage))
    axes(handles.compressed)
    imshow(uint8(fix(mrg)))
    axes(handles.decompressed)
    imshow(uint8(fix(Dmrg)))
    
    figure, imshow(getimage(handles.showimage)), imshow(uint8(fix(Dmrg))) 
    subplot(1,2,1),   imshow(getimage(handles.showimage)); subplot(1,2,2),imshow(uint8(fix(Dmrg))) 
    
   

    Ttime= etime(clock,Time);           % All time in Sec.
    Thour=fix(Ttime/3600);
    Tmp=round(rem(Ttime,3600));
    Tmin=fix(Tmp/60);
    Tsec=round(rem(Tmp,60));
            % elapsed time
    f=['Time: ',int2str(Thour),':',int2str(Tmin),''':',int2str(Tsec),''''''];
    set(handles.time,'String',f);



% --- Executes during object creation, after setting all properties.
function decompressed_CreateFcn(hObject, eventdata, handles)
% hObject    handle to decompressed (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate decompressed

 

% --- Executes during object creation, after setting all properties.
function originalimage_CreateFcn(hObject, eventdata, handles)
% hObject    handle to originalimage (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate originalimage


% --- Executes during object creation, after setting all properties.
function compressed_CreateFcn(hObject, eventdata, handles)
% hObject    handle to compressed (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate compressed



% --- Executes during object creation, after setting all properties.
function showimage_CreateFcn(hObject, eventdata, handles)
% hObject    handle to showimage (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% Hint: place code in OpeningFcn to populate showimage


% --- Executes during object creation, after setting all properties.
function compresstext_CreateFcn(hObject, eventdata, handles)
% hObject    handle to compresstext (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called

% --- Executes on button press in decompress.
function decompress_Callback(hObject, eventdata, handles)
% hObject    handle to decompress (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)


% --- Executes during object creation, after setting all properties.
function psnr_CreateFcn(hObject, eventdata, handles)
% hObject    handle to psnr (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function snr_CreateFcn(hObject, eventdata, handles)
% hObject    handle to snr (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function nmse_CreateFcn(hObject, eventdata, handles)
% hObject    handle to nmse (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function bitrateall_CreateFcn(hObject, eventdata, handles)
% hObject    handle to bitrateall (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function bitratecr_CreateFcn(hObject, eventdata, handles)
% hObject    handle to bitratecr (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% --- Executes during object creation, after setting all properties.
function time_CreateFcn(hObject, eventdata, handles)
% hObject    handle to time (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    empty - handles not created until after all CreateFcns called


% [In,n,h,m,nop,err,file_I2H,file_I2H_B,file_H2O,file_H2O_B]=ReadParams
%    Read input parametere from a file
%    return In=image matrix and
%        n=number of input nerons,  h=number of hidden nerons
%        m=number of output nerons, nop=number of input training patterns
%        err= error tolerance for terminatig network training
%        file...= files that contain weights of network

% veisi@ce.sharif.edu, Dec. 2003

function [In,n,h,m,nop,err,file_I2H,file_I2H_B,file_H2O,file_H2O_B]=ReadParams
    %file=input('Parameters file:');
    %im=input('Input image file:');
    Im = ('lena.bmp');%get(handles.showimage);
    In=imread(Im);
    %n=input('Input block size :');
    %h=input('compressed block size :');
    n=64;
    m=n;
    h=16;
    [p,q]=size(In);
    nop=p*q/n;
    err=1.0e-40;              % Error tolerance

    file_I2H='I2H.wgt';        % Input2Hidden weight
    file_H2O='H2O.wgt';        % Hidden2Output weight
    file_I2H_B='I2H_B.wgt';    % Input2Hidden bias weight vector
    file_H2O_B='H2O_B.wgt';
% [v,w,v_b,w_b]=ReadWeights(I,H):
%    Read weights of trained network with I nerons in Input and output layer
%                                      H nerons in hidden layer 
%    Return input2hidden layer weight matrix(v) and bias vector of hidden layer(v_b)
%        hidden2output layer weight matrix(w) and bias vector of output layer(w_b)

% veisi@ce.sharif.edu, Dec. 2003

function [v,w,v_b,w_b]=ReadWeights(I,H)

    file_I2H='I2H.wgt';        % Input2Hidden weight
    file_H2O='H2O.wgt';        % Hidden2Output weight
    file_I2H_B='I2H_B.wgt';    % Input2Hidden bias weight vector
    file_H2O_B='H2O_B.wgt';


    v=ReadFromFile(file_I2H,I,H);
    w=ReadFromFile(file_H2O,H,I);

    fid = fopen(file_I2H_B,'r+');
    v_b=fscanf(fid,'%f');
    fclose(fid);
    
    fid = fopen(file_H2O_B,'r+');
    w_b=fscanf(fid,'%f');
    fclose(fid);
    
    % Res=AddPattern(Mrg,Add,M,N,I):
%    Merg the blocks of image and return the result
%   adding the Ith block Add to Merg, M*N is the size of final image

% veisi@ce.sharif.edu, Dec. 2003

function Res=AddPattern(Mrg,Add,M,N,I)
    
    t=zeros(M,N);
    [M2,N2]=size(Add);  
       
    in=fix((I-1)*N2/N)*M2;
    jn=(mod((I-1)*N2,N));
   
    for i=in+1:in+N2
        for j=jn+1:jn+N2
            Mrg(i,j)=Add(i-in,j-jn);  
        end
    end 
    Res=Mrg;
    
function val=BitRate(BlockSize,NoOfBlocks,NoOfHiddenNeroun,NoOfBitsOut,NoOfBitsWeight)

% val=BitRate(BlockSize,NoOfBlocks,NoOfHiddenNeroun,NoOfBitsOut,NoOfBitsWeight)
% Return the value val of bit rate that is necessary for coding outputs of hidden
%        layer and weights from hidden to output layer,
%
% Parameters:
%   BlockSize= size of each block in pixel, i.e. 8*8=(64) 
%   NoOfBlocks= number of blocks of image=(all pixels of image)/BlockSize,
%   NoOfHiddenNeroun= number of neurons in hidden layer i.e. (16),
%   NoOfBitsOut= number of bits that require for coding hidden layer outputs i.e.integer(16)
%   NoOfBitsWeight= number of bits that require for coding hidden to output layer weights i.e.float(32)

% Hadi veisi,  March 2004

val= ((NoOfBlocks*NoOfHiddenNeroun*NoOfBitsOut)+(BlockSize*NoOfHiddenNeroun*NoOfBitsWeight))/(BlockSize*NoOfBlocks);

% o=f1(x):
%   Active function for Hidden layer nerons (sigmoid)

% veisi@ce.sharif.edu, March 2004

function o=f1(x)
    tmp1=1+exp(x.*-1);    
    o=(tmp1.^(-1));
    

    
    
% o=f1_p(x):
%    Derivative of Active function for Hidden layer nerons

% veisi@ce.sharif.edu, Dec. 2003

function o=f1_p(x)
    o=f1(x).*(1-f1(x));
    

    % [x]=NextPattern(Input,N,No):
%   [x]=NextPattern(Input,N,No)
%   Read next ith block of image Input by each block size=N
%   return x=input vector = t=target vector           
%
%   Assumed that Image(Input) is power of 2 in both dimentions.

% % veisi@ce.sharif.edu, Dec. 2003


function [x]=NextPattern(Input,N,No)
    tmp=sqrt(N);
    [i,j]=size(Input);
    i=i/tmp;
    j=j/tmp;
    in=fix((No-1)/i)*tmp;
    jn=(mod(No-1,i))*tmp;
    k=1;
    for i=in+1:in+tmp
        for j=jn+1:jn+tmp
            %x(i-in,j-jn)=Input(i,j); 
            x(k,1)=Input(i,j);  %(i-1)*tmp+j
            k=k+1;
        end
    end
  
 % Input=NextTrainPattern(P_Counter);
%   Return the next training pattern

% 
function Input=NextTrainPattern(P_Counter)
    
    tmp = mod(P_Counter,22);
    file=['TrainingSet\train' ,int2str(tmp),'lena.bmp'];
    Input=imread('lena.bmp');
    
% val=NMSE(Im,Im_hat):
%   Return the value val of Normalized-Mean-Square-Error (NMSE)
%        for image Im and it's compressed vertion Im_hat.

%  Hadi veiis, March 2004

function val=NMSE(Im,Im_hat)

[M,N]=size(Im);
tmp1=0;
tmp2=0;
for i=1:M
    for j=1:N
        tmp1=tmp1+(double(Im(i,j))-double(Im_hat(i,j)))^2;
        tmp2=tmp2+double(Im(i,j))^2;
    end
end
val=tmp1/tmp2;


% val=PSNR(Im,Im_hat):
%   Return the value val of Peak-Signal-to-Noise-Ratio(PSNR) in dB for 
%        the image Im and it's compressed version Im_hat,

% Hadi veisi, March 2004

function val=PSNR(Im,Im_hat)

[M,N]=size(Im);
tmp=0;
for i=1:M
    for j=1:N
        tmp=tmp+(double(Im(i,j))-double(Im_hat(i,j)))^2;
    end
end
tmp=tmp/(M*N);
val=10*log10(256^2/tmp);

% [Out]=ReadFromFile(FileName,Rows,Cols):
%   Read Rows*Cols data from FileName and return in Out,

% veisi@ce.sharif.edu, Dec. 2003

function [Out]=ReadFromFile(FileName,Rows,Cols)
    fid = fopen(FileName,'r+');
    Out=fscanf(fid,'%f');
    fclose(fid);

    tmp=zeros(Rows,Cols);
    k=1;
    for i=1:Rows
        for j=1:Cols
            tmp(i,j)=Out(k);
            k=k+1;
        end
    end
    Out=tmp;
    
 % Save_res(Vec,Fname):
%   Save the trained weights:
%       Vec=  Data
%       Fname=Output file name
%
% veisi@ce.sharif.edu, Dec. 2003

function Save_res(Vec,Fname)

    No=length(Vec);
    if No~=length(Fname)
        disp('Error! Number of data and output file name should be same.');
        return;
    end
    
    for i=1:No
        Data=Vec(i);
        file=Fname(i);
        file=[file,'.wgt'];
        fid = fopen(file,'w');
        if fid==-1
            disp('Error! cannot create the output file');
            return;
        end
        [M,N]=size(Data);
        
        %fprintf(fid,'The input to hidden layer:\n\n    ');
        for i=1:M
            for j=1:N
                fprintf(fid,'%6.2f    ',Data(i,j));
            end
            fprintf(fid,'\n');
        end
    
        fclose(fid);
    
    end %for

 
% Save_w(v,w,Name1,Name2)
% Save the trained weights:
%       v=encoder weights(input to hidden layer)
%       w=decoder weights(hidden to output layer)
%
function Save_w(v,w,Name1,Name2)
%    file=input('Output file name for encoding weights:');
%    save file 'v' -ASCII;
    file=Name1;
    file=[file,'.wgt'];
    fid = fopen(file,'w');
    if fid==-1
        disp('Error! cannot create the output file');
        return;
    end
    [M,N]=size(v);
    %str=['The input to hidden layer [',int2str(M),'*',int2str(N),'] :\n\n'];
    %fprintf(fid,str);
    for i=1:M
        for j=1:N
            fprintf(fid,'%10.6f    ',v(i,j));
        end
        fprintf(fid,'\n');
    end
    fclose(fid);
    
    %******************************************************************    
    file=Name2;
    file=[file,'.wgt'];
    fid = fopen(file,'w');
    
    if fid==-1
        disp('Error! cannot create the output file');
        return;
    end
    
    [M,N]=size(w);
    %str=['The hidden to output layer [',int2str(M),'*',int2str(N),'] :\n\n'];
    %fprintf(fid,str);
    for i=1:M
        for j=1:N
            fprintf(fid,'%10.6f    ',w(i,j));
        end
        fprintf(fid,'\n');
    end
    fclose(fid);

    
 
% Save_w(v,w,Name1,Name2)
% Save the trained weights:
%       v=encoder weights(input to hidden layer)
%       w=decoder weights(hidden to output layer)
%
function Save_w_Ver2(v,w,Name1,Name2)
%    file=input('Output file name for encoding weights:');
%    save file 'v' -ASCII;
    file=Name1;
    file=[file,'.wgt'];
    fid = fopen(file,'w');
    if fid==-1
        disp('Error! cannot create the output file');
        return;
    end
    [M,N,P]=size(v);
    %str=['The input to hidden layer [',int2str(M),'*',int2str(N),'] :\n\n'];
    %fprintf(fid,str);
    for i=1:M
        for j=1:N
            for k=1:P
                fprintf(fid,'%10.6f    ',v(i,j,k));
            end
            fprintf(fid,'\n');
        end
        fprintf(fid,'\n\n\n');
    end
    fclose(fid);
    
    %******************************************************************    
    file=Name2;
    file=[file,'.wgt'];
    fid = fopen(file,'w');
    
    if fid==-1
        disp('Error! cannot create the output file');
        return;
    end
    
    [M,N,P]=size(w);
    %str=['The hidden to output layer [',int2str(M),'*',int2str(N),'] :\n\n'];
    %fprintf(fid,str);
    for i=1:M
        for j=1:N
            for k=1:P
                fprintf(fid,'%10.6f    ',w(i,j));
            end
            fprintf(fid,'\n');
        end
        fprintf(fid,'\n\n\n');
    end
    fclose(fid);

    
 % val=SNR(Im,Im_hat)
%   Return the value val of Signal-to-Noise-Ratio(SNR)in dB 
%        for image Im and it's compressed vertion Im_hat.

%  Hadi veiis, March 2004

function val=SNR(Im,Im_hat)

tmp=NMSE(Im,Im_hat);
val=-10*log10(tmp);


function wait(times)

% wait procedure for specified time.
%	
%	WAIT(TIMES)
%	  TIMES - number of seconds (can be fractional).
%	Stops procedure for N seconds.
%	
% By hadi veisi [veisi@ce.sharif.edu]

if nargin ~= 1
  error('Usage: WAIT(TIMES)');
end

drawnow
t1 = clock;
while etime(clock,t1) < times,end;


