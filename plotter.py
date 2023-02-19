import matplotlib.pyplot as plt
import csv

if __name__ == '__main__':
    # episode,score,avg_score,max_score,loss,explore_ratio,duration
    with open('record.csv', 'r') as f:
        reader = csv.reader(f)
        first = True
        episode = []
        score = []
        avg_score = []
        max_score = []
        loss = []
        explore_ratio = []
        duration = []
        for row in reader:
            if first:
                episode_idx = row.index('episode')
                score_idx = row.index('score')
                avg_score_idx = row.index('avg_score')
                max_score_idx = row.index('max_score')
                loss_idx = row.index('loss')
                explore_ratio_idx = row.index('explore_ratio')
                duration_idx = row.index('duration')
                first = False
            else:
                episode.append(int(row[episode_idx]))
                score.append(float(row[score_idx]))
                avg_score.append(float(row[avg_score_idx]))
                max_score.append(float(row[max_score_idx]))
                loss.append(float(row[loss_idx]))
                explore_ratio.append(float(row[explore_ratio_idx]))
                duration.append(int(row[duration_idx]))

    plt.figure(figsize=(20, 10))
    plt.subplot(2, 3, 1)
    plt.plot(episode, score, label='score')
    plt.xlabel('episode')
    plt.ylabel('score')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(episode, avg_score, label='avg_score')
    plt.xlabel('episode')
    plt.ylabel('avg_score')
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.plot(episode, max_score, label='max_score')
    plt.xlabel('episode')
    plt.ylabel('max_score')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(episode, loss, label='loss')
    plt.xlabel('episode')
    plt.ylabel('loss')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(episode, explore_ratio, label='explore_ratio')
    plt.xlabel('episode')
    plt.ylabel('explore_ratio')
    plt.legend()

    plt.subplot(2, 3, 6)
    plt.plot(episode, duration, label='duration')
    plt.xlabel('episode')
    plt.ylabel('duration')
    plt.legend()

    plt.show()
