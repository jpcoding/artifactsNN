def sign_penalty(predicted, target, p =2):
    sign_diff = (predicted.sign() != target.sign()).float()
    return (sign_diff ** p).sum 