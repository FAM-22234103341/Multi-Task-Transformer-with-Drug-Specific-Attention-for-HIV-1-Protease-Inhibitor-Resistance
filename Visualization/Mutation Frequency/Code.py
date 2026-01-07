# 3. Mutation frequency across positions
mutation_freq = np.mean(sequence_matrix, axis=0)
axes[0,2].bar(range(1, 100), mutation_freq)
axes[0,2].set_xlabel('Position')
axes[0,2].set_ylabel('Mutation Frequency')
axes[0,2].set_title('Mutation Frequency Across Protease Positions')
