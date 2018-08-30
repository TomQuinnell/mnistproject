import numpy as np #importing libraries
from mnist import MNIST
import os
import sys
import pickle
import random
import pygame

#Setting up Pygame Constants and Objects
BLACK = (0,0,0) #sets up black RGB tuple
WHITE = (255,255,255) #sets up white RGB tuple
RED = (255,0,0) #sets up red RGB tuple
GREEN = (0,255,0) #sets up green RGB tuple
pygame.init() #starts up game engine module
pygame.display.set_caption("Poor Man's OCR system") #sets title
clock = pygame.time.Clock() #starts clock
SCREENSIZE = SCREENWIDTH, SCREENHEIGHT = 800, 600 #sets constants for window size
screen = pygame.display.set_mode(SCREENSIZE) #make windows load a game window


class Box:
    """Class for textboxes and buttons on screen, holds rect and text"""
    def __init__(self, rect, textDic, fontSize, thickness=3, colour=BLACK):
        self.rect = rect #position of box
        self.x, self.y, self.width, self.height = rect #position of box
        self.textDic = textDic #dictionary holding the text in this box, and the positions of the text
        self.fontSize = fontSize #size of text
        self.thickness = thickness #size of border
        self.colour = colour #colour of box
    def draw(self):
        pygame.draw.rect(screen, self.colour, self.rect, self.thickness) #draws rect
        for text in self.textDic.keys():
            pos = self.textDic[text]
            textDraw(str(text), pos,fontSize=self.fontSize) #draws the text in this box

class Net:
    """Class for Neural Network, holds weights and performs matrix calculations"""
    def __init__(self, inputSize, hiddenSize, outputSize):
        if os.path.isfile("weights.pkl"): #if weight pickle files exist
            weights = loadPickle("weights.pkl") #load weights
            self.weightsIH = weights[0] #set weights from input to hidden
            self.weightsHO = weights[1] #set weights from hidden to output
        else:
            self.weightsIH = 2 * np.random.random((inputSize, hiddenSize)) - 1 #weights from input layer to hidden layer
            self.weightsHO = 2 * np.random.random((hiddenSize, outputSize)) - 1 #weights from hidden layer to output layer
        self.learningRate = 0.1 #sets learning rate as 0.1

    def forwardMultiply(self,inputs):
        self.input = inputs
        self.hidden = np.dot(self.input, self.weightsIH) #multiplies input layer by weights (from input to hidden)
        self.hidden = self.sigmoid(self.hidden) #applies sigmoid to hidden layer
        self.output = np.dot(self.hidden, self.weightsHO) #multiplies hidden layer by weights (from hidden to output)
        self.output = self.sigmoid(self.output) #applies sigmoid to output layer
        return self.output

    def backpropagate(self,targets):
        outputErrors = np.subtract(targets, self.output)  #works out errors of the output layer
        derivHO = self.learningRate * self.output * (1 - self.output) #works out rate of change of errors
        derivHO = np.array([derivHO * outputErrors]) #works out rate of change of errors
        hiddenTranspose = np.transpose([self.hidden]) #transposes hidden layer matrix
        weightDeltasHO = np.dot(hiddenTranspose, derivHO) #works out values to add to weights
        self.weightsHO += weightDeltasHO #changes weights accordingly

        hiddenErrors = np.dot(outputErrors, self.weightsHO.transpose(),) #works out errors of the hidden layer
        derivIH = self.learningRate * self.hidden * (1 - self.hidden)  #works out rate of change of errors
        derivIH = np.array([derivIH * hiddenErrors])  #works out rate of change of errors
        inputTranspose = np.transpose([self.input])  #transposes hidden layer matrix
        weightDeltasIH = np.dot(inputTranspose, derivIH) #works out values to add to weights
        self.weightsIH += weightDeltasIH  #changes weights accordingly

    def sigmoid(self,x): #sigmoid function
        return 1 / (1 + np.exp(-x))


def textDraw(message,xypos,fontCol=BLACK,fontSize=30): #function to draw text onto screen
    fontName = pygame.font.match_font("arial") #loads arial font
    font = pygame.font.Font(fontName, fontSize) #loads arial font
    textSurface = font.render(message, True, fontCol) #renders this font
    screen.blit(textSurface,xypos) #renders this font onto screen

def drawScreen(boxList,clearScreen=True): #function to draw boxes and buttons onto screen
    if clearScreen: #if clear screen required, clear the screen
        screen.fill(WHITE)
    else: #(used in drawloop to keep current drawn digit)
        pygame.draw.rect(screen,WHITE,[0,0,800,97]) #draw a rect behind text to stop text overlapping
        pygame.draw.rect(screen,WHITE,[23,510,400,80]) #draw a rect behind text to stop text overlapping
    for box in boxList: #for every box on current screen, draw it
        box.draw()

def loadPickle(fileName): #function to load pickle file
    f = open(fileName, "rb")
    data = pickle.load(f)
    f.close()
    return data
def writePickle(fileName,data): #function to write pickle file
    f = open(fileName, "wb")
    pickle.dump(data, f)
    f.close()

def loadData(isTest=False): #function to load test data
    if isTest: #if in testing stage, set up testData filename
        fileName = "testData.pkl"
    else: #if in training stage, set up trainingData filename
        fileName = "trainingData.pkl"
    if os.path.isfile(fileName): #if pickle files already exist, load the data from these files
        data = loadPickle(fileName) #loads images and labels
        images = data[0] #images held at 0th index of data
        labels = data[1] #labels held at 1st index of data
        return images, labels
    else:
        mnistData = MNIST(os.getcwd()) #if pickle files don't exist, create instance of MNIST parser object at current directory
        if isTest:
            images,labels = mnistData.load_testing() #if in testing stage, make parser load test data
        else:
            images,labels = mnistData.load_training() #if in training stage, make parser load training data
        writePickle(fileName, (images,labels)) #writes this list to a pickle file to be loaded in later
        return images, labels


def trainNetwork(net):
    trainedNet, backPressed = training(net) #train net
    writePickle("weights.pkl", (trainedNet.weightsIH, trainedNet.weightsHO)) #save weights
    if backPressed:
        return trainedNet #if back button pressed in training loop, return to menu
    score, backPressed = testing(trainedNet)
    if backPressed:
        return trainedNet #if back button pressed in testing loop, return to menu
    percentSuccess = round(score / 1000 * 100, 1)
    frames = 0 #initialises frame count
    while frames <= 150: #for 2.5 seconds
        for event in pygame.event.get():
            if event.type == pygame.QUIT: #checks if red exit button has been pressed
                pygame.quit() #close game engine
                sys.exit() #exit
        screen.fill(WHITE)  #clear screen
        textDraw("Training Finished", [80,130], fontSize=100) #draw Training Finished message
        successMessage = "The network guessed " + str(percentSuccess) + "% of digits correctly"
        textDraw(successMessage, [25,275], fontSize=45)
        frames += 1 #add one to frame count
        pygame.display.flip() #renders to screen
        clock.tick(60) #sets loop to happen 60 times per second
    return trainedNet

def training(net):
    screen.fill(WHITE) #clear screen
    textDraw("Loading Training Data...", [60, 210], fontSize=80) #draw Loading screen
    pygame.display.flip()  #renders to screen
    clock.tick()  #renders to screen
    images, labels = loadData() #loads training data
    images, labels = images[:1000], labels[:1000]
    data = list(zip(images, labels)) #zip together images and labels
    random.shuffle(data) #shuffle data
    count = 0 #set count as 0
    boxList = [] #initialise list of boxes for screen
    titleBox = Box([0,0,0,0], {"Training Network...":[180,1]}, 70) #title box
    boxList.append(titleBox) #add to list
    digitBox = Box([23,98,424,424], {"Current Validation digit:":[25,62]}, 30) #digit box
    boxList.append(digitBox) #add to list
    guessBox = Box([470,98,320,424], {"Digit:":[475,98], "Network's":[475, 255], "guess:":[475,300]}, 60) #box for network's guesses
    boxList.append(guessBox) #add to list
    backButton = Box([470,535,320,55], {"Back":[580,527]}, 60) #back button
    boxList.append(backButton) #add to list
    drawScreen(boxList) #draw boxes onto screen
    for image,label in data: #for each image in training data:
        progressMessage = "Progress: " + str(count) + " of " + str(len(data)) #update progress message
        pygame.draw.rect(screen, WHITE, [25,525,440,100]) #draw rect behind progress message
        textDraw(progressMessage, [25,525], fontSize=45) #draw progress
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  #checks if red exit button has been pressed
                writePickle("weights.pkl",(net.weightsIH, net.weightsHO)) #save weights
                pygame.quit()  #close game engine
                sys.exit()  #exit
        image = np.array(image) #turns image into numpy array
        image = image / 255 #normalises grayscale values in image
        net.forwardMultiply(image) #multiply inputs through net
        if pygame.mouse.get_pressed()[0]: #if mouse button pressed
            mousePos = pygame.mouse.get_pos() #get mouse pos
            if mousePos[0] in range(backButton.x, backButton.x + backButton.width) and mousePos[1] in range(backButton.y, backButton.y + backButton.height):
                #if in back button rect
                return net, True #return net, with back button pressed
        if count % 500 == 0: #every 500th iteration
            drawScreen(boxList) #clears screen
            guess = np.argmax(net.output) #works out guess (highest output)
            drawDigit(image,guess,label) #draws digit and guess onto screen
        targets = np.zeros(10)  #set up targets
        targets[label] = 1
        net.backpropagate(targets) #adjust weights with these targets
        count += 1 #adds one to count
        pygame.display.flip() #renders to screen
        clock.tick() #sets loop to happen 60 times per second
    return net, False

def testing(net):
    screen.fill(WHITE)
    textDraw("Loading Testing Data...", [60, 210], fontSize=80)  #draw Loading screen
    pygame.display.flip()  #renders to screen
    clock.tick()  #renders to screen
    images, labels = loadData(isTest=True)  #loads training data
    images, labels = images[:1000], labels[:1000]
    data = list(zip(images, labels))  #zip together images and labels
    count = 0  #set count as 0
    netScore = 0 #sets score of network as 0
    boxList = []  #initialise list of boxes for screen
    titleBox = Box([0, 0, 0, 0], {"Testing Network...": [180, 1]}, 70)  #title box
    boxList.append(titleBox)  #add to list
    digitBox = Box([23, 98, 424, 424], {"Current Testing digit:": [25, 62]}, 30)  #digit box
    boxList.append(digitBox)  #add to list
    guessBox = Box([470, 98, 320, 424], {"Digit:": [475, 98], "Network's": [475, 255], "guess:": [475, 300]}, 60)  #box for network's guesses
    boxList.append(guessBox)  #add to list
    backButton = Box([470, 535, 320, 55], {"Back": [580, 527]}, 60)  #back button
    boxList.append(backButton)  #add to list
    drawScreen(boxList)  #draw boxes onto screen
    for image, label in data:  #for each image in training data:
        progressMessage = "Progress: " + str(count) + " of " + str(len(data))  #update progress message
        pygame.draw.rect(screen, WHITE, [25, 525, 440, 100])  #draw rect behind progress message
        textDraw(progressMessage, [25, 525], fontSize=45)  #draw progress
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  #checks if red exit button has been pressed
                pygame.quit()  #close game engine
                sys.exit()  #exit
        image = np.array(image)  #turns image into numpy array
        image = image / 255  #normalises grayscale values in image
        net.forwardMultiply(image)  #multiply inputs through net
        if pygame.mouse.get_pressed()[0]:  #if mouse button pressed
            mousePos = pygame.mouse.get_pos()  #get mouse pos
            if mousePos[0] in range(backButton.x, backButton.x + backButton.width) and mousePos[1] in range(backButton.y, backButton.y + backButton.height):
                #if in back button rect
                return netScore, True #return net, with back button pressed
        target = label #sets up target
        guess = np.argmax(net.output)  #works out guess (highest output)
        if target == guess:
            netScore += 1 #if guessed correctly, score a point
        if count % 500 == 0:  #every 500th iteration
            drawScreen(boxList)  #clears screen
            drawDigit(image, guess, target)  #draws digit and guess onto screen
        count += 1  #adds one to count
        pygame.display.flip()  #renders to screen
        clock.tick()  #sets loop to happen 60 times per second
    return netScore, False

def drawDigit(image, guess, target):
    image = np.resize(image,(28,28)) #resize image into 28 x 28 array
    startPos = [25,100] #start pos of drawing pad
    cellSize = 420//28 #size of each cell in grid
    for yPos, y in enumerate(image): #for each row in image
        for xPos, x in enumerate(y): #for each x in row
            pixelColour = 255 - x * 255 #un-normalises data, and turns into opposite colour
            colour = [pixelColour, pixelColour, pixelColour] #sets up grayscale value of colour
            pygame.draw.rect(screen, colour, [startPos[0] + xPos * cellSize, startPos[1] + yPos * cellSize, cellSize, cellSize]) #draw this rect in cell
    textDraw(str(target), [587,100], fontSize=60) #draw actual value of digit onto screen
    if guess == target: #if correct
        fontcolour = GREEN #make text green
    else:
        fontcolour = RED #make text red
    textDraw(str(guess), [632,302], fontCol=fontcolour, fontSize=60) #draw network's guess


def interface(net):
    boxList = []  #initialises list of boxes on screen
    background = Box([425, 0, 370, 600], {"Draw a Digit":[270, 1]}, 60, thickness=0, colour=WHITE)  #background box
    boxList.append(background)  #add to list
    drawingPad = Box([23, 98, 400, 400], {"Draw your number here:":[25, 60]}, 30)  #drawing pad box
    boxList.append(drawingPad)  #add to list
    guessBox = Box([440, 98, 347, 400], {"First Guess":[445, 105], "Second Guess":[445, 285]}, 30)  #box for network's guesses
    boxList.append(guessBox)  #add to list
    clearButton = Box([23, 510, 400, 80], {"Clear":[150, 510]}, 70)  #clear button
    boxList.append(clearButton)  #add to list
    backButton = Box([440, 510, 347, 80], {"Back":[560, 510]}, 70)  #back button
    boxList.append(backButton)  #add to list
    while True:
        drawScreen(boxList) #draws boxes onto screen
        option = "idle" #sets initial option as idle to enter the idle loop
        while option != "clear": #exits loop if option is clear, goes back to drawScreen() call which resets screen
            if option == "idle":
                option = idleLoop() #starts idle loop, and saves the option it returns
            elif option == "draw":
                option = drawLoop(net, boxList) #starts drawing loop, and saves the option it returns
            elif option == "back":
                return "back" #exits interface and returns back to menu screen

def idleLoop():
    drawingPad = Box([23,98,400,400], {"Draw your number here:":[25,60]}, 30) #drawing pad box
    clearButton = Box([23,510,400,80], {"Clear":[150,510]}, 70) #clear button
    backButton = Box([440,510,347,80], {"Back":[560,510]}, 70) #back button
    while True: #idle loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT: #checks if red exit button has been pressed
                pygame.quit() #close game engine
                sys.exit() #exit
        mousePos = pygame.mouse.get_pos() #gets mouse position
        if pygame.mouse.get_pressed()[0] == 1: #if left mouse button pressed
            if mousePos[0] in range(drawingPad.x + 20, drawingPad.x + drawingPad.width - 20) and mousePos[1] in range(drawingPad.y + 20, drawingPad.y + drawingPad.height - 20):
                #if mousepos in drawing pad
                return "draw" #go to drawing loop
            elif mousePos[0] in range(clearButton.x, clearButton.x + clearButton.width) and mousePos[1] in range(clearButton.y, clearButton.y + clearButton.height):
                #if mousepos in clear button
                return "clear" #start screen clear
            elif mousePos[0] in range(backButton.x, backButton.x + backButton.width) and mousePos[1] in range(backButton.y, backButton.y + backButton.height):
                #if mousepos in back button
                return "back" #exit interface loop
        pygame.display.flip() #renders to screen
        clock.tick(60) #sets loop to happen 60 times per second

def drawLoop(net, boxList):
    drawingPad = Box([23, 98, 400, 400], {"Draw your number here:":[25,60]}, 30)  #drawing pad box
    centrePos = [(drawingPad.x * 2 + drawingPad.width) // 2, (drawingPad.y * 2 + drawingPad.height) // 2] #centre of drawing pad
    furthest = [centrePos[0] - 20, centrePos[0] + 20, centrePos[1] - 20, centrePos[1] + 20] #used to keep track of furthest pixel left, right, up and down
    while True: #drawing loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT: #check if red exit button has been pressed
                pygame.quit() #close game engine
                sys.exit() #exit
        if pygame.mouse.get_pressed()[0] == 1: #if left mouse button pressed
            mousePos = pygame.mouse.get_pos() #gets mouse position
            if mousePos[0] in range(drawingPad.x + 20 , drawingPad.x + drawingPad.width - 20) and mousePos[1] in range(drawingPad.y + 20, drawingPad.y + drawingPad.height - 20):
                #if mouse position at least 20 pixels into drawing pad
                pygame.draw.circle(screen, BLACK, mousePos, 20) #draw a circle at current mouse position
                furthest = findFurthest(furthest, mousePos) #update furthest pixels
        else: #if left mouse button released
            inputs = convertToInputs(furthest) #convert pixels in drawing pad to 28x28 array to multiply through network
            outputs = net.forwardMultiply(inputs) #multiply this 28x28 array through trained network
            firstGuess, firstConfidence, secondGuess, secondConfidence = getConfidence(outputs) #get guesses and confidence
            drawScreen(boxList, clearScreen=False) #draw boxes onto screen but don't clear screen
            textDraw(str(firstGuess), [580,105]) #draw text of number guessed
            textDraw(str(firstConfidence) + "% confident", [445,135]) #draw text of confidence of this guess
            textDraw(str(secondGuess), [615,285]) #draw text of number guessed
            textDraw(str(secondConfidence) + "% confident", [445,320]) #draw text of confidence of this guess
            return "idle" #go back to idle loop
        pygame.display.flip() #renders to screen
        clock.tick(60) #sets loop to happen 60 times per second

def findFurthest(furthest, mousePos):
    left,right,up,down = furthest #current furthest pixels in all directions
    newFurthest = furthest #sets up new furthest pixels as original values
    if mousePos[0] - 20 < left: #if left pixel of circle further left
        newFurthest[0] = mousePos[0] - 20 #update furthest left pixel
    if mousePos[0] + 20 > right: #if right pixel of circle further right
        newFurthest[1] = mousePos[0] + 20 #update furthest right pixel
    if mousePos[1] - 20 < up: #if highest pixel of circle further up
        newFurthest[2] = mousePos[1] - 20 #update furthest up pixel
    if mousePos[1] + 20 > down: #if lowest pixel of circle further down
        newFurthest[3] = mousePos[1] + 20 #update furthest down pixel
    return newFurthest

def convertToInputs(furthest):
    pixelArray = [] #array used to store colour values of pixels
    for y in range(100, 500): #for y coordinate from 100 to 500 (drawing pad position)
        rowList = [] #list used to store colour values of pixels in certain row
        for x in range(25,425): #for x coordinate from 25 to 425 (drawing pad position)
            if screen.get_at([x,y])[:-1] == BLACK: #if colour of pixel == black
                rowList.append(1) #append 1 to list (representing 100% darkness)
            else:
                rowList.append(0) #append 0 to list (representing 0% darkness)
        pixelArray.append(rowList) #add the list of colours in the row to the overall array
    pixelArray = resizeArray(furthest, pixelArray) #re-sizes array down to wrap around digit drawn
    cellSize = len(pixelArray) // 20 #sets size for single cell (1/20th of drawing pad size)
    averageArray = [] #array that holds average greyscale value
    for border in range(28 * 4):
        averageArray.append(0) #adds 'border' of white pixels to centralise digit in array
    for yMult in range(20):
        for border in range(4):
            averageArray.append(0) #adds 'border' of white pixels to centralise digit in array
        for xMult in range(20):
            cellList = [] #list used to hold colours of current cell
            for row in pixelArray[yMult * cellSize:(yMult+1) * cellSize - 1]: #narrows down y pixels to those with a Y coordinate in current cell
                xPixels = row[xMult * cellSize:(xMult + 1) * cellSize - 1] #narrows down x pixels to those with an X coordinate in current cell
                for colour in xPixels:
                    cellList.append(colour) #individually adds colour of each pixel in current cell
            averageArray.append(np.mean(cellList)) #add mean of grayscale values to array after colour values of current cell collected
        for border in range(4):
            averageArray.append(0) #adds 'border' of white pixels to centralise digit in array
    for border in range(28*4):
        averageArray.append(0) #adds 'border' of white pixels to centralise digit in array
    return averageArray #returns 28x28 array, ready to multiply through network

def resizeArray(furthest, digit):
    padding = 25 #sets up number of pixels in border around digit (used to prevent furthest going beyond drawing pad range - NaN error)
    left, right, up, down = furthest #furthest pixels in all directions
    digit = np.pad(digit, padding, 'edge') #creates border around edge of digit
    left += padding - 23 #transposes value of furthest left pixel to new padded digit
    right += padding - 23 #transposes value of furthest right pixel to new padded digit
    up += padding - 97 #transposes value of furthest up pixel to new padded digit
    down += padding - 97 #transposes value of furthest down pixel to new padded digit
    xDifference = right - left #difference in left and right furthest pixel positions
    yDifference = down - up #difference in up and down furthest pixel positions
    centre = [(left + right) // 2, (up + down) // 2] #finds centre of furthest pixels
    newWidth = np.max((xDifference, yDifference)) // 2 #works out new array width and height (largest difference in x and y directions)
    digit = digit[centre[1] - newWidth: centre[1] + newWidth] #narrows down rows in digit
    resizedDigit = [] #initialises array to store resized digit
    for row in digit:
        resizedDigit.append(row[centre[0] - newWidth: centre[0] + newWidth]) #squish every row in the digit
    resizedDigit = np.array(resizedDigit) #convert to numpy array
    return resizedDigit

def getConfidence(outputs):
    totalConfidence = np.sum(outputs)  #finds total confidence, used to calculate relative confidence of guesses
    firstGuess = np.argmax(outputs)  #gets index of largest value in outputs
    firstConfidence = int(outputs[firstGuess] / totalConfidence * 100)  #converts the value (from 0 to 1) at this index to a percentage value and compares to overall confidence
    outputs[firstGuess] = 0  #sets largest value to 0 to find next biggest value
    secondGuess = np.argmax(outputs)  #gets index of next largest value in outputs
    secondConfidence = int(outputs[secondGuess] / totalConfidence * 100)  #converts the value (from 0 to 1) at this index to a percentage value and compares to overall confidence
    if firstConfidence == 0:
        firstConfidence = 2  #prevents the chance of network seeming to be 0% sure
    if secondConfidence == 0:
        secondConfidence = 1  #prevents the chance of network seeming to be 0% sure
    return firstGuess, firstConfidence, secondGuess, secondConfidence


def menu():
    net = Net(784, 256, 10) #creates instance of network (input size 784, hidden size 256, output size 10)
    boxList = [] #initialises list of boxes on screen
    titleBox = Box([0,0,0,0], {"Poor Man's OCR System":[85,3]}, 70) #title box
    boxList.append(titleBox) #add to list
    trainingButton = Box([150,100,500,125], {"Train Network":[250,125]}, 60) #button to enter training loop
    boxList.append(trainingButton) #add to list
    drawingButton = Box([150,230,500,125], {"Draw a digit":[275,255]}, 60) #button to enter drawing loop
    boxList.append(drawingButton) #add to list
    exitButton = Box([150,360,500,125], {"Exit":[360,387]}, 60) #exit button
    boxList.append(exitButton) #add to list
    drawScreen(boxList) #draw boxes onto screen
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: #check if red exit button has been pressed
                pygame.quit() #close game engine
                sys.exit() #exit
        if pygame.mouse.get_pressed()[0]: #if left mouse button pressed
            mousePos = pygame.mouse.get_pos() #get mouse position
            if mousePos[0] in range(trainingButton.x, trainingButton.x + trainingButton.width): #if x pos of mouse in button rects
                if mousePos[1] in range(trainingButton.y, trainingButton.y + trainingButton.height): #if mouse y pos in training button rect
                    net = trainNetwork(net) #start training
                    drawScreen(boxList) #draw menu screen
                elif mousePos[1] in range(drawingButton.y, drawingButton.y + drawingButton.height): #if mouse y pos in drawing button rect
                    interface(net) #start drawing interface loop
                    drawScreen(boxList) #draw menu screen
                elif mousePos[1] in range(exitButton.y, exitButton.y + exitButton.height): #if mouse y pos in exit button rect
                    pygame.quit() #close game engine
                    sys.exit() #exit
        pygame.display.flip() #renders to screen
        clock.tick(60) #sets loop to happen 60 times per second

menu()