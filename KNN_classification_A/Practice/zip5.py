players = ["Aman", "Bina", "Chetan", "Divya"]
scores = [55, 88, 72, 90]

players_sorted, scores_sorted = (
    list(t) for t in zip(*sorted(zip(players, scores), key=lambda x: x[1], reverse=True))
)

print("players_sorted =", players_sorted)
print("scores_sorted =", scores_sorted)