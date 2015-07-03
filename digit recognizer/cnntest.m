  function [test_y] = cnntest(net, x)
    %  feedforward
    net = cnnff(net, x);
    [~, test_y] = max(net.o);
end
