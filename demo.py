from nn import MLP

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

model = MLP(3, [4, 4, 1])

for epoch in range(100):

    # forward pass
    predictions = [model(x) for x in xs]

    loss = sum(
        (prediction - target) ** 2
        for prediction, target in zip(predictions, ys)
    )

    # reset gradients
    for p in model.parameters():
        p.grad = 0.0

    # backward pass
    loss.backward()

    # update
    learning_rate = 0.05
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    if epoch % 10 == 0:
        print("Epoch:", epoch, "Loss:", loss.data)

print("\nFinal predictions:")
for x, target in zip(xs, ys):
    prediction = model(x)
    print(
        "Input:", x,
        "| Predicted:", round(prediction.data, 3),
        "| Actual:", target
    )