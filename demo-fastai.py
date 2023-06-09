import streamlit as st
import pandas as pd
from fastai.tabular.all import *
from fastai.collab import *

@st.cache_data
def load_data():
    data_df = pd.read_excel('[final] April 2015 to Nov 30 2019 - Transformed Jester Data - .xlsx',
                            header=None,
                            names=None,
                            index_col=None)

    jokes_df = pd.read_excel('Dataset4JokeSet.xlsx',
                             header=None,
                             names=None,
                             index_col=None)

    data_df.drop(0, axis=1, inplace=True)

    data_df.columns = jokes_df.index

    data_df = data_df.stack().reset_index()
    data_df.columns = ['user_id', 'joke_id', 'rating']

    data_df = data_df.loc[data_df['rating'] != 99]

    jokes_df.columns = ['joke']
    jokes_df.index.name = 'joke_id'

    return data_df, jokes_df

def train_model(data_df):
    dls = CollabDataLoaders.from_df(ratings=data_df,
                                    item_name='joke_id',
                                    user_name='user_id',
                                    rating_name='rating',
                                    valid_pct=0.1,
                                    bs=256, )
    learn = collab_learner(dls,
                           n_factors=20,
                           y_range=(-10, 10),
                           use_nn=True,
                           loss_func=None
                           )
    learn.fit_one_cycle(5, 0.001, 0.1)

    return learn


def recommend_jokes(learn, data_df, jokes_df, new_user_id, new_ratings):
    # Convert ratings from 0-5 scale to -10 to 10 scale
    new_ratings = {joke_id: info['rating'] * 4 - 10 for joke_id, info in new_ratings.items()}

    # Add new user's ratings to the data
    new_ratings_df = pd.DataFrame({
        'user_id': [new_user_id] * len(new_ratings),
        'joke_id': list(new_ratings.keys()),
        'rating': list(new_ratings.values()),
        'joke': jokes_df.loc[list(new_ratings.keys()), 'joke'].values
    })

    data_df = pd.concat([data_df, new_ratings_df])

    # Generate recommendations for the new user
    joke_ids = data_df['joke_id'].unique()  # Get the list of all joke ids

    joke_ids_new_user = data_df.loc[
        data_df['user_id'] == new_user_id, 'joke_id']  # Get the list of joke ids rated by the new user

    joke_ids_to_pred = np.setdiff1d(joke_ids, joke_ids_new_user)  # Get the list of joke ids the new user has not rated

    # Predict the ratings for all unrated jokes
    testset_new_user = pd.DataFrame({
        'user_id': [new_user_id] * len(joke_ids_to_pred),
        'joke_id': joke_ids_to_pred,
        'joke': jokes_df.loc[joke_ids_to_pred, 'joke'].values
    })

    test_dl = learn.dls.test_dl(testset_new_user)
    preds, _ = learn.get_preds(dl=test_dl)

    # Add predictions to the testset_new_user DataFrame
    testset_new_user['rating'] = preds.numpy()

    # Get the top 5 jokes with highest predicted ratings
    top_5_jokes = testset_new_user.nlargest(5, 'rating')

    return top_5_jokes


def main():
    data_df, jokes_df = load_data()

    new_user_id = data_df['user_id'].max() + 1

    if 'initial_ratings' not in st.session_state:
        st.session_state.initial_ratings = {}
        random_jokes = jokes_df.sample(3)  # 随机选取3条
        for joke_id, joke in zip(random_jokes.index, random_jokes['joke']):
            st.session_state.initial_ratings[joke_id] = {'joke': joke, 'rating': 3}

    with st.form(key='initial_ratings_form'):
        for joke_id, info in st.session_state.initial_ratings.items():
            st.write(info['joke'])
            info['rating'] = st.slider('Rate this joke', 0, 5, step=1, value=info['rating'], key=f'rec_{joke_id}')

        if st.form_submit_button('Submit Ratings'):
            # Train model
            learn = train_model(data_df)

            # Recommend jokes based on user's ratings
            recommended_jokes = recommend_jokes(learn, data_df, jokes_df, new_user_id, st.session_state.initial_ratings)

            # Save recommended jokes to session state
            st.session_state.recommended_jokes = {}
            for joke_id, joke in zip(recommended_jokes['joke_id'], recommended_jokes['joke']):
                st.session_state.recommended_jokes[joke_id] = {'joke': joke, 'rating': 3}

    if 'recommended_jokes' in st.session_state:

        st.write('We recommend the following jokes based on your ratings:')

        with st.form(key='recommended_ratings_form'):
            # 显示基于用户评分所推荐的笑话
            for joke_id, info in st.session_state.recommended_jokes.items():
                st.write(info['joke'])
                info['rating'] = st.slider('Rate this joke', 0, 5, step=1, value=info['rating'], key=f'rec_{joke_id}')

            if st.form_submit_button('Submit Recommended Ratings'):
                ratings = [info['rating'] for joke_id, info in st.session_state.recommended_jokes.items()]
                total_score = sum(ratings)
                percentage_of_total = (total_score / 25) * 100
                st.write(f'You rated the recommended jokes {percentage_of_total}% of the total possible score.')


if __name__ == '__main__':
    main()
