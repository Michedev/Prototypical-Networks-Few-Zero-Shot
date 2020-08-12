from operator import itemgetter

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage


class Tester:

    def __init__(self, model, loss_f, device):
        self.model = model
        self.loss_f = loss_f
        self.device = device

    def calc_loss(self, distances, y_test):
        distances = distances.reshape(distances.size(0) * distances.size(1), distances.size(2))
        y_test = y_test.flatten()
        return self.loss_f(distances, y_test)

    def calc_accuracy(self, distances, y_test):
        distances = distances.reshape(distances.size(0) * distances.size(1), distances.size(2)).argmax(-1)
        y_test = y_test.flatten(1)
        return (distances == y_test).float().mean()

    def test_step(self, engine, batch):
        X_train, y_train = batch["train"]
        X_test, y_test = batch["test"]
        X_train = X_train.to(self.device); X_test = X_test.to(self.device)
        y_train = y_train.to(self.device); y_test = y_test.to(self.device)
        distances = self.model(X_train, y_train, X_test)
        loss = self.calc_loss(distances, y_test)
        acc = self.calc_accuracy(distances, y_test)
        return loss, acc

    def test(self, test_loader, test_len=None):
        tester = Engine(self.test_step)

        @tester.on(Events.EPOCH_STARTED)
        def init_vars(engine):
            engine.state.sum_loss = 0.0
            engine.state.sum_acc = 0.0

        @tester.on(Events.ITERATION_COMPLETED)
        def accumulate(engine):
            loss, acc = engine.state.output
            engine.state.sum_loss += loss
            engine.state.sum_acc += acc

        @tester.on(Events.EPOCH_COMPLETED)
        def view_output(engine: Engine):
            mean_loss = engine.state.sum_loss / engine.state.iteration
            mean_acc = engine.state.sum_acc / engine.state.iteration
            print('Test loss', mean_loss, '-', 'Test accuracy', mean_acc)

        RunningAverage(output_transform=itemgetter(0)).attach(tester, 'test_loss')
        RunningAverage(output_transform=itemgetter(1)).attach(tester, 'test_accuracy')

        ProgressBar().attach(tester, ['test_loss', 'test_accuracy'])

        tester.run(test_loader, 1, test_len)

