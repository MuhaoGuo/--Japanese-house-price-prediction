from sklearn.linear_model import LinearRegression


def Linear_Regression_Algorithm(X_train_0A,
                                Y_train_0A,
                                X_train_AB,
                                Y_train_AB,
                                X_train_BC,
                                Y_train_BC,
                                X_train_CD,
                                Y_train_CD,
                                X_test,
                                Y_test,
                                predict_class
                                ):

    lr_0A = LinearRegression().fit(X_train_0A, Y_train_0A)
    lr_AB = LinearRegression().fit(X_train_AB, Y_train_AB)
    lr_BC = LinearRegression().fit(X_train_BC, Y_train_BC)
    lr_CD = LinearRegression().fit(X_train_CD, Y_train_CD)

    result = []
    print(".......................")
    print(X_test)

    for i in range(len(X_test)):
        pre_c = predict_class[i]
        x_t = X_test[i:i+1].values

        # print("x_t is " + str(x_t))
        # print("result i is " + str(lr_0A.predict(x_t)))

        if pre_c == "0 ~ A":
            result.append(lr_0A.predict(x_t)[0])
        elif pre_c == "A ~ B":
            result.append(lr_AB.predict(x_t)[0])
        elif pre_c == "B ~ C":
            result.append(lr_BC.predict(x_t)[0])
        elif pre_c == "C ~ +":
            result.append(lr_CD.predict(x_t)[0])

    print("linear regression predict result is" + str(result))

    from sklearn.metrics import r2_score
    score = r2_score(Y_test, result)

    print("linear regression score is " + str(score))





