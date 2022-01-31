from __future__ import division
from __future__ import print_function

import sys
import cv2
import editdistance
import numpy as np
sys.path.insert(0, 'drive/My Drive/Licenta/Cod/WBS/')


from DataLoader import DataLoader,Batch
from spellchecker import SpellChecker

from WordBeamSearch import wordBeamSearch
from LanguageModel import LanguageModel
import codecs

class Trainer:

    corpus = "drive/My Drive/Licenta/Data/dateH5/IAM/corpus.txt"
    wordCharList = "drive/My Drive/Licenta/Data/dateH5/IAM/wordCharList.txt"

    def __init__(self, accuracyPath, charList, spellCheck = False):

        self.accuracyPath = accuracyPath
        self.charList = charList
        self.spellCheck = spellCheck


        if self.spellCheck:
            self.checker = SpellChecker()
            self.checker.word_frequency.load_text_file(Trainer.corpus)

        chars = "".join(self.charList)
        wordChars = codecs.open(
            Trainer.wordCharList, 'r', 'utf8').read()
        corpus = codecs.open(Trainer.corpus, 'r', 'utf8').read()
        self.languageModel = LanguageModel(corpus, chars, wordChars)


    def train(self,model,loader):

        epoch = 0
        bestCharErrorRate = float('inf')
        noImprovementsSince = 0
        earlyStopping = 8

        while True:

            epoch += 1
            print('Epoch', epoch)
            print('Train NN')
            loader.trainSet()
            while loader.hasNext():
                iterInfo = loader.getIteratorInfo()
                batch = loader.getNext()
                loss = model.trainBatch(batch)
                if iterInfo[0] % 10 == 0:
                	print('Batch:', iterInfo[0], '/', iterInfo[1], 'Loss:', loss)

            charErrorRate,accuracy,wordErrorRate = self.validate(model, loader)

            # if best validation accuracy so far, save model parameters
            if charErrorRate < bestCharErrorRate:
                print('Character error rate improved, save model')
                bestCharErrorRate = charErrorRate
                noImprovementsSince = 0
                model.save()
                open(self.accuracyPath, 'w').write('Validation character error rate of saved model: %f%%' % (charErrorRate * 100.00))
            else:
                print('Character error rate not improved')
                noImprovementsSince += 1

            if noImprovementsSince >= earlyStopping:
                print('No more improvements since %d epochs. Training stopped.' % earlyStopping)
                break

    def validate(self,model, loader):

        print('Validate NN')
        loader.validationSet()
        numCharErr = 0
        numCharTotal = 0
        numLineOk = 0
        numLineTotal = 0

        totalCER = []
        totalWER = []

        while loader.hasNext():

            iterInfo = loader.getIteratorInfo()
            if iterInfo[0] % 10 == 0:
            	print('Batch:', iterInfo[0], '/', iterInfo[1])
            batch = loader.getNext()
            recognized,_ = model.inferBatch(batch)

            #print('Ground truth -> Recognized')
            for i in range(len(recognized)):

                numLineOk += 1 if batch.groundTexts[i] == recognized[i] else 0
                numLineTotal += 1

                distance = editdistance.eval(recognized[i], batch.groundTexts[i])
                currCER = distance / max(len(recognized[i]), len(batch.groundTexts[i]))
                totalCER.append(currCER)

                currWER = self.wer(recognized[i].split(), batch.groundTexts[i].split()) / max(len(recognized[i].split()),
                                                                                              len(batch.groundTexts[i].split()))
                totalWER.append(currWER)

                numCharErr += distance
                numCharTotal += len(batch.groundTexts[i])
                #if iterInfo[0] % 10 == 0:
                #    print('[OK]' if distance == 0 else '[ERR:%d]' % distance, '"' +
                #    batch.groundTexts[i] + '"', '->', '"' + recognized[i] + '"')

        charErrorRate = sum(totalCER)/ len(totalCER)
        accuracy = numLineOk / numLineTotal
        wordErrorRate = sum(totalWER) / len(totalWER)

        print('Character error rate: %f%%. Address accuracy: %f%%. Word error rate: %f%%' %
              (charErrorRate * 100.0, accuracy * 100.0, wordErrorRate * 100.0))
        return charErrorRate, accuracy, wordErrorRate

    def infer(self,model, images):

        batch =  Batch(None, images)
        recognized,_ = model.inferBatch(batch, True)
        return recognized

    def score(self, model, batch, useWBS = False):

        numCharErr = 0
        numCharTotal = 0
        numLineOk = 0
        numLineTotal = 0

        totalCER = []
        totalWER = []
        recognized, ctcOutput = model.inferBatch(batch)

        if useWBS:
            recognized = []
            for i in range(ctcOutput.shape[1]):
                mat = ctcOutput[:,i,:]
                recognized.append(wordBeamSearch(mat, 25, self.languageModel, True))

        for i in range(len(recognized)):
            numLineOk += 1 if batch.groundTexts[i] == recognized[i] else 0
            numLineTotal += 1

            if self.spellCheck:
                words = recognized[i].split()
                for j in range(len(words)):
                    words[j] = self.checker.correction(words[j])
                recognized[i] = ' '.join(word for word in words)

            distance = editdistance.eval(recognized[i], batch.groundTexts[i])
            currCER = distance / max(len(recognized[i]), len(batch.groundTexts[i]))
            totalCER.append(currCER)

            currWER = self.wer(recognized[i].split(), batch.groundTexts[i].split()) / max(
                len(recognized[i].split()), len(batch.groundTexts[i].split()))
            totalWER.append(currWER)

            numCharErr += distance
            numCharTotal += len(batch.groundTexts[i])

        charErrorRate = sum(totalCER)/ len(totalCER)
        accuracy = numLineOk / numLineTotal
        wordErrorRate = sum(totalWER) / len(totalWER)

        print('Character error rate: %f%%. Address accuracy: %f%%. Word error rate: %f%%' %
              (charErrorRate * 100.0, accuracy * 100.0, wordErrorRate * 100.0))
        return charErrorRate, accuracy, wordErrorRate

    def wer(self,r, h):

        d = np.zeros((len(r)+1)*(len(h)+1), dtype=np.uint8)
        d = d.reshape((len(r)+1, len(h)+1))
        for i in range(len(r)+1):
            for j in range(len(h)+1):
                if i == 0:
                    d[0][j] = j
                elif j == 0:
                    d[i][0] = i

        # computation
        for i in range(1, len(r)+1):
            for j in range(1, len(h)+1):
                if r[i-1] == h[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    substitution = d[i-1][j-1] + 1
                    insertion = d[i][j-1] + 1
                    deletion = d[i-1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)
        return d[len(r)][len(h)]