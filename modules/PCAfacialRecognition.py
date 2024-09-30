import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from sklearn.metrics import  confusion_matrix, accuracy_score, precision_score, recall_score
import seaborn as sns


class PCAfacialRecognition:
    
    def __init__(self, datasetPath = ''):
        # attributes
        self.datasetPath = datasetPath
        self.numPeople = 40
        self.numFaces = 10
        self.classes = np.arange(0, 40+1)
        # attributes of the images
        self.heightImage = (plt.imread(datasetPath+"archive/s1/1.pgm")).shape[0]
        self.widthImage = (plt.imread(datasetPath+"archive/s1/1.pgm")).shape[1]
        self.numPixels = self.heightImage * self.widthImage
   
    
    def DatasetSplit(self, seed: int, numPeopleTraining: int, numFacesTraining: int):
        """
        The method splits the dataset in traing set and test set.

        Parameters
        ----------
        seed : int
            A seed to feed the random number generator.
        numPeopleTraining : int
            Number of people selected for the training phase.
        numFacesTraining : int
            Number of images selected for every person in the training set.
            
        New attributes
        -------
        numPeopleTraining : int
            Number of people selected for the training phase.
        numFacesTraining : int
            Number of images selected for each person in the training set.
        numImagesTraining : int
            Number of images present in the training set.
        numImagesTest : int
            Number of images present in the test set.
        peopleTraining : array
            Array that contains the people selected for the training.
        peopleTest : array
            Array that contains the people selected for the test. The algorithm should classify them as "not in the database".
        TrainingMat : array
            A matrix containing the "vectorized" images selected for the training phase.
        train_people : list
            It records who is depicted in every image of the training set.
        train_face : list
            For every person in the training set, the list records which images are selected
        TestMat : array
            A matrix containing the "vectorized" images selected for the test phase.
        test_people : list
            It records who is depicted in every image of the test set.
        test_face : list
            For every person in the test set, the list records which images are selected
        """
        
        listPeople = list(np.arange(1, self.numPeople+1))
        listFaces = list(np.arange(1, self.numFaces+1))
        
        self.numPeopleTraining = numPeopleTraining
        self.numFacesTraining = numFacesTraining
        self.numImagesTraining = self.numPeopleTraining * self.numFacesTraining
        self.numImagesTest = self.numPeople * self.numFaces - self.numImagesTraining
        self.TrainingMat = []
        self.train_people = []
        self.train_face = []
        self.TestMat = []
        self.test_people = []
        self.test_face = []
        
        rng = np.random.default_rng(seed)
        rng.shuffle(listPeople)

        # people selected for the trainning
        self.peopleTraining = []
        for person in (listPeople.copy())[:self.numPeopleTraining]:
            self.peopleTraining.append(person)
            rng.shuffle(listFaces)
            # images selected for the training
            for face in (listFaces.copy())[:self.numFacesTraining]:
                im = np.array(plt.imread(self.datasetPath +'archive/s{}/{}.pgm'.format(str(person), str(face))), dtype='float64')
                self.TrainingMat.append(im.reshape(self.numPixels))
                self.train_people.append(person)
                self.train_face.append(face)
            # images selected for the test
            for face in (listFaces.copy())[self.numFacesTraining:self.numFaces]:
                im = np.array(plt.imread(self.datasetPath +'archive/s{}/{}.pgm'.format(str(person), str(face))), dtype='float64')
                self.TestMat.append(im.reshape(self.numPixels))
                self.test_people.append(person)
                self.test_face.append(face)
         
        # people that were not selceted for the training are added to the test set and should be not be recognized by the algorithm
        self.peopleTest = []
        for person in (listPeople.copy())[self.numPeopleTraining:self.numPeople]:
            self.peopleTest.append(person)
            for face in listFaces:
                im = np.array(plt.imread(self.datasetPath +'archive/s{}/{}.pgm'.format(str(person), str(face))), dtype='float64')
                self.TestMat.append(im.reshape(self.numPixels))
                self.test_people.append(person)
                self.test_face.append(face)
                
        # lists are transformed into matrices
        self.TrainingMat = np.array(self.TrainingMat).T        
        self.TestMat = np.array(self.TestMat).T
        
    def PrincipalComponents(self):
        """
        The method returns the principal components( also called "eigenfaces"), that are the eigenvectors of the empirical covariance matrix. 
    
        New attributes
        -------
        meanFace: array
            "Vectorized" empirical mean of the images of the training set.
        TranslatedTrainingMat: array
             Matrix that contains the translated images of the training set (the "mean face" is substracted from each image).
        rankA: int
            Rank of the auxiliary covariance matrix
        orderedEigenvaluesCovMat : array
            Array containing the absolute value of the eigenvalues in decreasing order.
        
        eigenFaces : bidemnsional array
            Principal components or eigenfaces.
            
        meanFace: array
            Empirical mean of the images of the training set.
    
        """
        # calculate the mean face
        self.meanFace = np.reshape(np.mean(self.TrainingMat, axis=1), self.numPixels)
        
        # translate the dataset with respect to the mean
        self.TranslatedTrainingMat = np.zeros((self.numPixels, self.numImagesTraining))
        for image in range(self.numImagesTraining):    
            self.TranslatedTrainingMat[:, image] = self.TrainingMat[:, image] - self.meanFace
        
        
        # calculate the auxiliary matrix
        A = (self.TranslatedTrainingMat.T).dot(self.TranslatedTrainingMat) 
        eigenvaluesA, eigenvectorsA = sc.linalg.eig(A)
        self.rankA = np.linalg.matrix_rank(A)
        
        # sorting eigenvalues in descending order based on their magnitude
        absEigenValuesA = np.absolute(eigenvaluesA.copy())
        indices = np.argsort(absEigenValuesA)[::-1] # indices[i] = j <-> the i-th largest eigenvalue by magnitude was located at position j
        self.orderedEigenvaluesA = np.sort(absEigenValuesA)[::-1]

        # sorting eigenvectors according to the order of the eigenvalues
        orderedAeigenvectors = np.zeros((self.numImagesTraining, self.rankA))
        for eigenvector in range(self.rankA):
            orderedAeigenvectors[:,eigenvector] = eigenvectorsA[:, indices[eigenvector]]
        
        # calculate the eigenvectors of the empirical covariance matrix
        self.eigenFaces = np.matmul(self.TranslatedTrainingMat, eigenvectorsA)
        
        # normalize the eigenvectors of the empirical covariance matrix
        for eigenvector in range(self.rankA):
            self.eigenFaces[:, eigenvector] = self.eigenFaces[:, eigenvector]/np.linalg.norm(self.eigenFaces[:, eigenvector])
   
        
    def SubspaceBasis(self, variancePercentage):
         """
         The method returns a orthonormal basis of the subspace generated by the eigenfaces that retains at least the variance percenatge desired. 
    
         Parameters
         ----------
         variancePercentage : float
             Percentage of variance the user wants to retain.
         
         Returns
         -------------
         subspaceDimension : int
             Dimension of the subspace generated.
         subspaceBasis : array
             Basis of the subspace generated.
         
         """  
         if variancePercentage < 1 and variancePercentage > 0:
             totalVariance = np.linalg.norm(self.orderedEigenvaluesA, 1)
             self.subspaceDimension = 0
             variancePreserved = 0
             
             while(variancePreserved < variancePercentage * totalVariance):
                 variancePreserved += self.orderedEigenvaluesA[self.subspaceDimension]
                 self.subspaceDimension  += 1
         elif variancePercentage == 1:
             self.subspaceDimension = self.rankA
         else:
             print("Percentage of variance must be STRICTLY greater than zero and less than 1")
             return -1
         
         # basis of the subspace     
         self.subspaceBasis = self.eigenFaces[:, :self.subspaceDimension]
         
         return self.subspaceDimension, self.subspaceBasis 
    
        
    def TrainingSetProjection(self):
        """
        The method projects the images of the training dataset into the subspace generated by the eigenfaces.

        New attributes
        --------------
        Coefficients: array
            Matrix containing the dot product among the image of the dataset and the eigenfaces. These are the coordinates of the images with respect to the basis of the subspace.
        
        """
        self.Coefficients = np.matmul(self.TranslatedTrainingMat.T, self.subspaceBasis)
        
    
    def TrainingSetReconstruction(self):
        """
        The function reconstructs the images starting from their coordinates in the subspace generated by the eigenfaces.

        Returns
        -------
        ReconstructedImages : array
            Matrix that contains the reconstructed images
        
        """
        
        self.ReconstructedImages = np.matmul(self.subspaceBasis, self.Coefficients.T)
        for image in range(self.numImagesTraining):
            self.ReconstructedImages[:,image] = self.ReconstructedImages[:, image] + self.meanFace
            
        
        return self.ReconstructedImages
    
    def FacialRecognition(self, TestImage, threshold, showImPred = False):
        """
        Given an image and a threshold for the error, the function predicts which individual from the training set is depicted. 
        Specifically, the prediction is based on the training image that is closest to the given image. 
        If the minimum distance exceeds the specified threshold, the function indicates that the image does not contain an individual from the training set.

        Parameters
        ----------
        TestImage : array
            Image.
        threshold : float
            threshold for the error.
        showImPred : bool, optional
            The default value is False. If it is set equal to one, the function plots the closest image.

        Returns
        -------
        found : bool
            It is equal to one if the function identifies who is depicted in the given image.
        person : int
            It identifies the person depicted in the given image.
        face : int
            It identifies which among the images of the identified person is the closest to the given image.
            
        New attributes
        --------------
        error : array
            array that contains the distance between the test image and each image of the training set.

        """    
        # ensure that the image is vectorized
        TestImageVec = np.reshape(TestImage.copy(), self.numPixels)
        
        # error[i] = 2 norm of the difference between the input image and the i-th image of the dataset.
        self.error = np.zeros(self.numImagesTraining)
        
        # search for the minimum error
        err_min = np.inf
        for image in range(self.numImagesTraining):
            #error[image] = np.linalg.norm(TestImageVec - self.ReconstructedImages[:, image])
            self.error[image] = np.linalg.norm(np.matmul(self.subspaceBasis.T, (TestImageVec - self.meanFace)) - self.Coefficients[image, :])
            if self.error[image] < err_min:
                err_min = self.error[image]
                person = self.train_people[image]
                face = self.train_face[image]
                
        if err_min > threshold:
            found = 0
            person = -1
            face = -1
            if showImPred == True:
                print("The image provided contains an individual that is not in the dataset")
                
        else:
            found = 1
            if showImPred == True:
                print("\nThe image provided belongs to {}".format(person))
                print("The closest image is {}".format(face))
                plt.imshow(plt.imread(self.datasetPath+'archive/s{}/{}.pgm'.format(str(person), str(face))), cmap='gray')
                plt.title("Closest image found in the database")
                plt.axis('off')
                plt.show()    
            
        return found, person, face
           
    def ExecuteTest(self, thresholdError, showImTest=False, showImPred=False, showConfMat=False):
        """
        The function executes the test using the test set.

        Parameters
        ----------
        thresholdError : float
            Threshold that determines if the person is identified or not.
        showImTest : bool, optional
            The default is False. If it is set equal to one, each image in the test set is plotted.
        showImPred : bool, optional
            The default is False. If it is set equal to one, each predicted image is plotted.
        showConfMat : bool, optional
            The default is False. If it is set equal to one, the confusion matrix is plotted.

        Returns
        -------
        accuracy : float
            Percentage of images that are correctly classified.
        precision : array 
            Array containg the precision score for each class.
        recall : array
            Array containg the recall score for each class.
        confusionMatrix : array
            Confusion matrix. 
        
        New attributes
        --------------
        predicted_label : array
            Array that contains the prediction of the function.
        predicted_face : 
            Array that contains which of ten images of the predicted individual is the closest to the given image.
        """
        
        self.predicted_label = []
        self.predicted_face =  []
        for immagine in range(self.TestMat.shape[1]):
            # visualizzazione immagine test
            if showImTest == True:
                person = self.test_people[immagine]
                face =  self.test_face[immagine]
                plt.imshow(np.reshape(self.TestMat[:,immagine], (self.heightImage,self.widthImage)), cmap='gray')
                plt.title("Person {} Face {}".format(int(person), int(face)))
                plt.axis('off')
                plt.show()
            found, person, face = self.FacialRecognition(self.TestMat[:,immagine], thresholdError, showImPred)
            if found == 1:
                self.predicted_label.append(person)
                self.predicted_face.append(face)
            else:
                # the image is labelled with a zero if the function can not attribute the given image to an individual in the training set.
                self.predicted_label.append(0)
                
        self.accuracy = accuracy_score(self.test_people, self.predicted_label)
        self.precision = precision_score(self.test_people, self.predicted_label, average = None, zero_division=0)
        self.recall = recall_score(self.test_people, self.predicted_label, average = None)
        self.confusionMatrix = confusion_matrix(self.test_people, self.predicted_label)
        
        if showConfMat == True: 
            plt.figure(figsize=(10, 8)) 
            sns.heatmap(self.confusionMatrix, annot=False, cmap="Blues",fmt="d")
            plt.xlabel("Predicted Label")
            plt.ylabel("True Label")
            plt.title("Confusion Matrix")
            plt.show()

        return self.accuracy, self.precision, self.recall, self.confusionMatrix  
    


