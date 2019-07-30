


for t in range(0, 5):
    #
    input_batch, target_batch = next(iter(training_loader))
    #
    input_batch  = input_batch.to('cuda')    # the shape of input_batch = (batch_size, 1, FREQ_BAND, SOUND_WINDOW)
    target_batch = target_batch.to('cuda')   # the shape of target_batch = (batch_size)
    #
    output_batch = classifier(input_batch)
    sm = F.softmax(output_batch, dim=1)
    #
    total1 = 0
    total2 = 0
    success_rate1 = 0
    success_rate2 = 0
    #
    correct=[]
    wrong=[]
    for t in range(0, 50):
        v, i = torch.max(sm[t], 0)
        if target_batch[t]==i:
            correct.append(t)
        else:
            wrong.append(t)
        #
        total1 += 1
    #
    success_rate1 += len(correct)
    print(len(correct), len(wrong))
    #    
    correct=[]
    wrong=[]
    for t in range(0, 50):
        v, i = torch.topk(sm[t], 2) 
        if len((i==target_batch[t]).nonzero())>0:
            correct.append(t)
        else:
            wrong.append(t)
        #
        total2 += 1
    #
    success_rate2 += len(correct)
    print(len(correct), len(wrong))
    #
    print(success_rate1/total1, success_rate2/total2)


#----------------------------------------------------------------------------------------------

idx=92

print("original and fitted amplitutde//frequency:")          
x = [i  for i in range(0, 240)]
plt.plot(x, inputs[candidates][:, 0:1, :, 5:6][idx][0,:,0].cpu().detach().numpy())
plt.plot(x, audios[idx][0,:,0].cpu().detach().numpy())
plt.show()            

print("fitted amplitutde//frequency:")   
x = [i  for i in range(0, 240)]
plt.plot(x, audios[idx][0,:,0].cpu().detach().numpy())
plt.show()



#----------------------------------------------------------------------------------------------

# load the audio generator model
#
classifierPath = 'C:\\Python\\MachineLearning\\NeuralNetwork\\CNN\\pytorch model\\'
classifierName = 'Classifier_Inception_TIMIT_DR1.pt'
classifierModelPathName = classifierPath + classifierName
# load the trained classifier neural network
classifier = Classifier().cuda()
classifier.load_state_dict(torch.load(classifierModelPathName))
classifier.eval()
    
generatorPath = 'C:\\Python\\MachineLearning\\NeuralNetwork\\CNN\\pytorch model\\'
generatorName = 'Generator_Inception_TIMIT.pt'
generatorModelPathName = generatorPath + generatorName
# load the trained classifier neural network
generator = AudioGenerator().cuda()
generator.load_state_dict(torch.load(generatorModelPathName))
generator.eval()
    
softmaskPath = 'C:\\Python\\MachineLearning\\NeuralNetwork\\CNN\\pytorch model\\'
softmaskName = 'SoftMask_Inception_TIMIT.pt'
softmaskModelPathName = softmaskPath + softmaskName
# load the trained classifier neural network
softmask = AudioGenerator().cuda()
softmask.load_state_dict(torch.load(softmaskModelPathName))
softmask.eval()


# load a sample audio file
pathName = "C:\\Data\\TIMIT\\Data\\TRAIN\\DR8\\MBCG0\\"
wavfileName = "SI957.WAV.wav" 
wavFilePathName = pathName + wavfileName
#
freqBandCnt=240   # number of frequency bands to produce from fast Fourier transform 
(frame_size, frame_stride) = (0.030, 0.010)  # audio frame size and frame stride in seconds
channel, spectrogram, phasegram, frqcyBnd, sample_rate, sample_cnt = FourierTransform.FFT(wavFilePathName, 0, frame_size, frame_stride, duration=0, emphasize=False)
#
(_, _, frameCnt) = np.shape(spectrogram)
init = True
spect=None
#
spectrogram = torch.as_tensor(spectrogram, dtype=torch.float)



# run each fram through the generator and cumulate the generated freq/amp file
for f in range(15, frameCnt-15):
    #
    inMatrix = classifierInput(spectrogram, f)
    inMatrix = inMatrix.expand(1, -1,-1,-1)   # make it a batch of one data piece by expanding one dimension
    #
    output = classifier(inMatrix)
    fm = classifier.featureMatrix
    audio = generator(fm)
    #
    audio = audio.squeeze(dim=0)
    if init:
        spect=audio
        init=False
    else:
        spect = torch.cat((spect, audio), dim=2)


# generate a wave file
generatedWavFilePath = 'C:\\Python\\MachineLearning\\NeuralNetwork\\CNN\\pytorch model\\'
generatedWavFilePathName = generatedWavFilePath + wavfileName
FourierTransform.playSound(frame_size, frame_stride, spect, None, frqcyBnd, sample_rate, 0, 3, generatedWavFilePathName)

    
#----------------------------------------------------------------------------------------------------    
#----------------------------------------------------------------------------------------------------    
    
idx=11

print("weight and mask comparison:")          
x = [i  for i in range(0, 240)]
plt.plot(x, weight[idx, 0, :, 0].cpu().detach().numpy())
plt.plot(x, mask[idx, 0, :, 0].cpu().detach().numpy())
plt.show()            


#----------------------------------------------------------------------------------------------------    

pathName = "C:\\Data\\TIMIT\\Data\\TRAIN\\DR1\\MTRR0\\"
audioFile = "SI918.WAV"

freqBandCnt=240   # number of frequency bands to produce from fast Fourier transform 
(frame_size, frame_stride) = (0.030, 0.010)  # audio frame size and frame stride in seconds

# load a sample audio file
wavfileName = audioFile + ".wav" 
wavFilePathName = pathName + wavfileName
channel, spect_origin, phasegram, frqcyBnd, sample_rate, sample_cnt, fft = FourierTransform.FFT(wavFilePathName, 0, frame_size, frame_stride, duration=0, emphasize=False)
spect_origin = torch.as_tensor(spect_origin, dtype=torch.float)
#spect_noise = spect_noise.to('cuda')

# load a sample noisy audio file
wavfileName = audioFile + ".noise.wav" 
wavFilePathName = pathName + wavfileName
#
channel, spect_noise0, phasegram, frqcyBnd, sample_rate, sample_cnt, fft = FourierTransform.FFT(wavFilePathName, 0, frame_size, frame_stride, duration=0, emphasize=False)
spect_noise0 = torch.as_tensor(spect_noise0, dtype=torch.float)
#spect_noise = spect_noise.to('cuda')
#
(_, _, frameCnt) = np.shape(spect_noise0)


MARGIN=15
init = True
spect_cleaned = None


spect_noise = spect_noise0


frameSeq=[]

# add the margin frames on the left; these frames are not going to be 'cleaned'.
for i in range(0, MARGIN): frameSeq.append(spect_noise[:,:,i:i+1])

# run each fram through the generator and cumulate the generated freq/amp file
for f in range(MARGIN, frameCnt-MARGIN):
    #
    inMatrix_noise = classifierInput(spect_noise, f)
    inMatrix_noise = inMatrix_noise.expand(1, -1,-1,-1)   # make it a batch of one data piece by expanding one dimension
    #inMatrix_noise = inMatrix_noise.to('cuda')
    #
    output_noise = classifier(inMatrix_noise)
    fm = classifier.featureMatrix
    mask = softmask(fm)
    #
    mask = torch.squeeze(mask, dim=0)
    frame_noise = spect_noise[:, :, f:f+1]
    #frame_noise = frame_noise.to('cuda')
    frame_cleaned = mask * frame_noise
    frameSeq.append(frame_cleaned)
    #
    print("process frame: ", f, "/", frameCnt)
    #torch.cuda.empty_cache()
    #print("cuda cached memory= ", torch.cuda.memory_cached())
#

# add the margin frames on the right; these frames are not going to be 'cleaned'.
# total number of frames are exactly the same as the noisy frame sequence.
for i in range(frameCnt-MARGIN, frameCnt): frameSeq.append(spect_noise[:,:,i:i+1])

# concatenate all frames into a pytorch tensor
spect_cleaned = torch.cat(frameSeq, dim=2)



soundWAVUtil.spectrogramHeatmap(spect_origin.detach()[0], frqcyRange=[0, 150], ampRange=[0, 0.06])
soundWAVUtil.spectrogramHeatmap(spect_cleaned.detach()[0], frqcyRange=[0, 150], ampRange=[0, 0.06])
soundWAVUtil.spectrogramHeatmap(spect_noise0.detach()[0], frqcyRange=[0, 150], ampRange=[0, 0.06])
soundWAVUtil.spectrogramHeatmap(spect_noise.detach()[0], frqcyRange=[0, 150], ampRange=[0, 0.06])
soundWAVUtil.spectrogramHeatmap(spect_cleaned0.detach()[0], frqcyRange=[0, 150], ampRange=[0, 0.06])
soundWAVUtil.spectrogramHeatmap(spect_cleaned1.detach()[0], frqcyRange=[0, 150], ampRange=[0, 0.06])


# multiple passes
spect_cleaned0 = spect_cleaned.clone()
spect_cleaned1 = spect_cleaned.clone()
spect_noise = spect_cleaned
(_, _, frameCnt) = np.shape(spect_noise)



path = 'C:\\Python\\MachineLearning\\NeuralNetwork\\CNN\\pytorch model\\'
spectName = 'spect_cleaned7.pt'
torch.save(spect_cleaned, path+spectName)
spect_cleaned=None
spect_noise = torch.load(path+spectName)

spect_cleaned = torch.load(path+'spect_cleaned6.pt')


pO=[]
pN=[]
pC=[]

for f in range(MARGIN, frameCnt-MARGIN):
    #
    inMatrix_origin = classifierInput(spect_origin, f)
    inMatrix_origin = inMatrix_origin.expand(1, -1,-1,-1)   # make it a batch of one data piece by expanding one dimension
    #
    output_origin = classifier(inMatrix_origin)
    phone_origin = classifierToPhone(output_origin)
    pO.append(phone_origin)
    #
    #
    inMatrix_noise = classifierInput(spect_noise0, f)
    inMatrix_noise = inMatrix_noise.expand(1, -1,-1,-1)   # make it a batch of one data piece by expanding one dimension
    #
    output_noise = classifier(inMatrix_noise)
    phone_noise = classifierToPhone(output_noise)
    pN.append(phone_noise)
    #
    #
    inMatrix_cleaned = classifierInput(spect_cleaned, f)
    inMatrix_cleaned = inMatrix_cleaned.expand(1, -1,-1,-1)   # make it a batch of one data piece by expanding one dimension
    #
    output_cleaned = classifier(inMatrix_cleaned)
    phone_cleaned = classifierToPhone(output_cleaned)
    pC.append(phone_cleaned)
    #
    #
    print("process frame: ", f, "/", frameCnt)
#




pO = np.array(pO)
pN = np.array(pN)
pC = np.array(pC)

ON=(pO==pN)
OC=(pO==pC)

print(OC.sum(), ON.sum())

#----------------------------------------------------------------------------------------------------    


L1 = 0
L2 = 0
for f in range(15, frameCnt-15):
    origin = spectrogram_original[:, :, f:f+1]
    wNoise = spectrogram[:, :, f:f+1]
    masked = spect_cpu[:, :, f-15:f-15+1]
    #
    l1 = scipy.spatial.distance.cdist(origin[0], wNoise[0]).sum()
    l2 = scipy.spatial.distance.cdist(origin[0], masked[0]).sum()
    #
    L1 += l1
    L2 += l2
    #
    print("frame improvement=", 1-l2/l1, "L1=", L1, "L2=", L2)
#
    

# generate a wave file
generatedWavFilePath = 'C:\\Python\\MachineLearning\\NeuralNetwork\\CNN\\pytorch model\\'
wavfileName = "SX327.WAV.cleaned.wav"
generatedWavFilePathName = generatedWavFilePath + wavfileName
FourierTransform.playSound(frame_size, frame_stride, spect_cpu, phasegram, frqcyBnd, sample_rate, 0, 5, generatedWavFilePathName)


#----------------------------------------------------------------------------------------------------  

# run each fram through the generator and cumulate the generated freq/amp file
for f in range(15, frameCnt-15):
    #
    inMatrix_noise = classifierInput(spect_noise, f)
    inMatrix_noise = inMatrix_noise.expand(1, -1,-1,-1)   # make it a batch of one data piece by expanding one dimension
    #
    output_noise = classifier(inMatrix_noise)
    #
    fm = classifier.featureMatrix
    mask = softmask(fm)
    #
    mask = torch.squeeze(mask, dim=0)
    frame_noise = spect_noise[:, :, f:f+1]
    frame_cleaned = mask * frame_noise
    #
    if init:
        spect=frame_cleaned
        init=False
    else:
        spect_cleaned = torch.cat((spect_cleaned, frame_cleaned), dim=2)
    #   
    torch.cuda.empty_cache()
    print("cuda cached memory= ", torch.cuda.memory_cached())
#


    

spect_cpu = spect.cpu().detach()
spect_cpu = np.array(spect_cpu)

#----------------------------------------------------------------------------------------------------    
#----------------------------------------------------------------------------------------------------    


x = np.arange(5)
ft  = scipy.fftpack.fft(x)
ift = scipy.fftpack.ifft(ft)
np.allclose(ift, x, atol=1e-15)  # within numerical accuracy.




            
def classifierInput(spectrogram, fIdx):
    #
    c0=torch.cat((spectrogram[:,:, fIdx-5 :fIdx-5+1], 
                  spectrogram[:,:, fIdx-4 :fIdx-4+1],
                  spectrogram[:,:, fIdx-3 :fIdx-3+1],
                  spectrogram[:,:, fIdx-2 :fIdx-2+1],
                  spectrogram[:,:, fIdx-1 :fIdx-1+1],
                  spectrogram[:,:, fIdx-0 :fIdx-0+1],
                  spectrogram[:,:, fIdx+1 :fIdx+1+1],
                  spectrogram[:,:, fIdx+2 :fIdx+2+1],
                  spectrogram[:,:, fIdx+3 :fIdx+3+1],
                  spectrogram[:,:, fIdx+4 :fIdx+4+1],
                  spectrogram[:,:, fIdx+5 :fIdx+5+1]), dim=2)
    #
    c1=torch.cat((spectrogram[:,:, fIdx-10:fIdx-10+1], 
                  spectrogram[:,:, fIdx-8 :fIdx-8+1],
                  spectrogram[:,:, fIdx-6 :fIdx-6+1],
                  spectrogram[:,:, fIdx-4 :fIdx-4+1],
                  spectrogram[:,:, fIdx-2 :fIdx-2+1],
                  spectrogram[:,:, fIdx-0 :fIdx-0+1],
                  spectrogram[:,:, fIdx+2 :fIdx+2+1],
                  spectrogram[:,:, fIdx+4 :fIdx+4+1],
                  spectrogram[:,:, fIdx+6 :fIdx+6+1],
                  spectrogram[:,:, fIdx+8 :fIdx+8+1],
                  spectrogram[:,:, fIdx+10:fIdx+10+1]), dim=2)
    #
    c2=torch.cat((spectrogram[:,:, fIdx-15:fIdx-15+1], 
                  spectrogram[:,:, fIdx-12:fIdx-12+1],
                  spectrogram[:,:, fIdx-9 :fIdx-9+1],
                  spectrogram[:,:, fIdx-6 :fIdx-6+1],
                  spectrogram[:,:, fIdx-3 :fIdx-3+1],
                  spectrogram[:,:, fIdx-0 :fIdx-0+1],
                  spectrogram[:,:, fIdx+3 :fIdx+3+1],
                  spectrogram[:,:, fIdx+6 :fIdx+6+1],
                  spectrogram[:,:, fIdx+9 :fIdx+9+1],
                  spectrogram[:,:, fIdx+12:fIdx+12+1],
                  spectrogram[:,:, fIdx+15:fIdx+15+1]), dim=2)
    #
    inMatrix=torch.cat((c0,c1,c2), dim=0)
    #
    return inMatrix
#

           
def classifierToPhone(classOut):
    softmax = F.softmax(classOut, dim=1)
    vlu, idx = torch.max(softmax[0], 0)
    phone = phonecode[idx-1][1]
    return phone
    