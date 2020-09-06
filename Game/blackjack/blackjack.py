import util



class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's hand.
    # The second element is the index (not the value) of the next card, if the player peeked in the last action.
        # If they didn't peek, this will be None.
    # The final element is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be done in succAndProbReward
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges coming out of |state|.
    # Indicate a terminal state (after quitting or busting) by setting the deck to None.
    # When the probability is 0 for a particular transition, don't include that in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE
        succ_prob_reward_list = []
        card_sum, peek_idx, deck = state  # card_sum = the sum of taken cards' values

        if deck is None:  # when there is no card in the deck
            pass          # no possible successor state

        elif action == 'Take':
            num_all_cards = sum(deck)  # the number of all cards

            # get_succ_reward(idx) returns a successor state and a reward, when a card is taken.
            def get_succ_reward(idx):
                new_card_sum = card_sum + self.cardValues[idx]  # HINT: use self.cardValues  # what's the new sum of card values, when we take a new card?
                if new_card_sum > self.threshold:  # when the card sum exceeds the threshold
                    new_deck = None
                    reward = 0
                elif num_all_cards > 1:  # sum(new_deck) > 0; when some cards remain
                    deckList = list(deck)
                    # decrease the number of instances of the taken card.
                    deckList[idx] -=1 
                    new_deck = tuple(deckList)
                    reward = 0
                else:  # when there is no card remaining
                    new_deck = None
                    reward = new_card_sum
                succ = new_card_sum, None, new_deck
                return succ, reward

            # Peek implementation ----------------------------------------
            if peek_idx is not None:  # when previous action was 'Peek'
                succ, reward = get_succ_reward(peek_idx)
                succ_prob_reward_list.append((succ, 1, reward))
            # ---------------------------------------- Peek implementation
            else:  # when previous action was not 'Peek'
                for idx, num in enumerate(deck):
                    if num == 0:
                        continue
                    succ, reward = get_succ_reward(idx)
                    prob = num/sum(deck)
                    succ_prob_reward_list.append((succ, prob, reward))

        # Peek implementation ----------------------------------------
        elif action == 'Peek':
            if peek_idx is None:
                num_all_cards = sum(deck)

                for idx, num in enumerate(deck):
                    if num == 0:
                        continue
                    prob = num/sum(deck) 
                    succ_prob_reward_list.append(((card_sum, idx, deck) , prob, - self.peekCost))  # HINT: has the form (new_card_sum, new_peek_idx, new_deck)
        # ---------------------------------------- Peek implementation

        elif action == 'Quit':
            succ_prob_reward_list.append(((card_sum, None, None), 1, card_sum))

        else:
            raise ValueError("Undefined action '{}'".format(action))

        return succ_prob_reward_list
        # END_YOUR_CODE

    def discount(self):
        return 1


if __name__ == '__main__':
    mdp = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

    algorithm = util.ValueIteration()
    algorithm.solve(mdp, verbose=0)

    for s in algorithm.pi:
        print(f'pi({s}) = {algorithm.pi[s]}')
