import numpy as np

def overlapScore(rects1, rects2):
    avgScore = 0
    scores = []

    for i, _ in enumerate(rects1):
        rect1 = rects1[i]
        rect2 = rects2[i]

        left = np.max((rect1[0], rect2[0]))
        right = np.min((rect1[0]+rect1[2], rect2[0]+rect2[2]))
        top = np.max((rect1[1], rect2[1]))
        bottom = np.min((rect1[1]+rect1[3], rect2[1]+rect2[3]))

        i = np.max((0, right-left)) * np.max((0, bottom-top))

        u = rect1[2]*rect1[3] + rect2[2]*rect2[3] - i

        score = np.clip(i/u, 0, 1)
        avgScore += score
        scores.append(score)

    return avgScore, scores
