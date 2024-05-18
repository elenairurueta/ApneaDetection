from Imports import *

'''
#Scheduler: programador de tasa de aprendizaje

def train_one_epoch(model, data_loader, optimizer, scheduler):
    model.train() #modo entrenamiento
    total_loss = 0

    for batch_index, data in enumerate(data_loader):
        loss = train_one_step(model, data, optimizer)
        scheduler.step()
        total_loss += loss
    return total_loss

def train_one_step(model, data, optimizer):
    optimizer.zero_grad()
    loss = model(data[0], data[1]) #'**data' desempaqueta un diccionario en datos y labels
    loss.backward() #se calculan los gradientes de la pérdida respecto a los parámetros del modelo 
    optimizer.step() #se actualizan los parámetros del modelo utilizando el optimizador
    return loss

def validate_one_epoch(model, data_loader):
    model.eval() #modo de evaluación
    total_loss = 0

    for batch_index, data in enumerate(data_loader):
        with torch.no_grad():
            loss = validate_one_step(model, data)
        total_loss += loss

    return total_loss

def validate_one_step(model, data):
    loss = model(data[0], data[1])  #'**data' desempaqueta un diccionario en datos y labels
    return loss
'''

#The Dataset is responsible for accessing and processing single instances of data.
#The DataLoader pulls instances of data from the Dataset collects them in batches, and returns them for consumption by your training loop.


def Train1(model, trainset, testset):
    n_epochs = 100
    loader = DataLoader(trainset, shuffle=True, batch_size=32)

    X_test, y_test = default_collate(testset)
    loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01) #se puede agregar momentum=0.9

    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        y_pred = model(X_test)
        acc = (y_pred.round() == y_test).float().mean()
        print(f"End of epoch {epoch}: accuracy = {float(acc)*100:.2f}%")

#Hiperparámetros: batch_size, loss_fn, optimizer


def Train2(model, trainset, valset):
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(valset, batch_size=4, shuffle=False)

    loss_fn = nn.CrossEntropyLoss() #BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1) #se puede agregar momentum=0.9

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    epoch_number = 0
    EPOCHS = 5
    best_vloss = 1_000_000.

    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))
    
        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_loss = train_one_epoch(epoch_number, train_loader, optimizer, loss_fn, model)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()

        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for i, vdata in enumerate(validation_loader):
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))


        # Track best performance, and save the model's state
       # if avg_vloss < best_vloss:
       #     best_vloss = avg_vloss
       #     model_path = 'model_{}_{}.pt'.format(timestamp, epoch_number)
       #     torch.save(model.state_dict(), model_path)

        epoch_number += 1


def train_one_epoch(epoch_index, train_loader, optimizer, loss_fn, model):
    running_loss = 0.
    last_loss = 0.
    
    for i, data in enumerate(train_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.

    return last_loss


def Train3(model, trainset, testset):

    train_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)


    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    start_time = time.time()
    epochs = 5
    train_losses = []
    test_losses = []
    train_correct = []
    test_correct = []

    #TRAIN
    for i in range(epochs):
        trn_corr = 0
        tst_corr = 0

        for idx, (X_train, y_train) in enumerate(train_loader):
            idx +=1 #start batches at 1
            y_pred = model(X_train) #predicted values for the training set
            loss = criterion(y_pred, y_train) #compare predictions to correct answers

            predicted = torch.max(y_pred.data, 1)[1] #add up the number of correct predictions
            trn_corr += (predicted == y_train).sum() #keep track of how many are correct in this batc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if idx%10 == 0: #imprimir cada 10
                print(f'Epoch: {i}. Batch: {idx}. Loss: {loss.item()}')
            
        train_losses.append(loss)
        train_correct.append(trn_corr)

    #TEST
    with torch.no_grad(): #no gradient so we don't update our weights and biases
        for idx, (X_test, y_test) in enumerate (test_loader):
            y_val = model(X_test)
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()

    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(tst_corr)


    current_time = time.time()
    total = current_time - start_time
    print(f'Training took {total/60}min')


'''
cost = torch.nn.CrossEntropyLoss()

# used to create optimal parameters
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Create the training function

def train(dataloader, model, loss, optimizer):
    model.train()
    size = len(dataloader.dataset)
    for batch, (X, Y) in enumerate(dataloader):
        
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        pred = model(X)
        loss = cost(pred, Y)
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')


# Create the validation/test function

def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for batch, (X, Y) in enumerate(dataloader):
            X, Y = X.to(device), Y.to(device)
            pred = model(X)

            test_loss += cost(pred, Y).item()
            correct += (pred.argmax(1)==Y).type(torch.float).sum().item()

    test_loss /= size
    correct /= size

    print(f'\nTest Error:\nacc: {(100*correct):>0.1f}%, avg loss: {test_loss:>8f}\n')
    '''
'''
epochs = 15
for t in range(epochs):
    print(f'Epoch {t+1}\n-------------------------------')
    train(train_dataloader, model, cost, optimizer)
    test(test_dataloader, model)
print('Done!')
'''
'''
model.eval()
test_loss, correct = 0, 0
class_map = ['no', 'yes']

with torch.no_grad():
    for batch, (X, Y) in enumerate(test_dataloader):
        X, Y = X.to(device), Y.to(device)
        pred = model(X)
        print("Predicted:\nvalue={}, class_name= {}\n".format(pred[0].argmax(0),class_map[pred[0].argmax(0)]))
        print("Actual:\nvalue={}, class_name= {}\n".format(Y[0],class_map[Y[0]]))
        break
'''
