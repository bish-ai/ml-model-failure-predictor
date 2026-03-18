{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPtFRqjtYLs61H64goY4S3M",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/bish-ai/ml-model-failure-predictor/blob/main/ml_model_failure_predictor.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "mPZTbH6iDh9Z"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "import seaborn as sb\n",
        "import matplotlib.pyplot as plt\n",
        "dl_data=pd.read_csv(\"https://storage.googleapis.com/kagglesdsdata/datasets/4819714/8149575/machine_failure_cleaned.csv?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20260318%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20260318T174527Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=bdb5935b3b2efacd08dada958fac2f424bfb08497388d4281e568874c5ac699e9130d16ffbd96e88b688321d0f64818c43251d05a9976a7c86ea4e301aab804a718558c3301160d202fed59a0cb83de92ba8bd16cb52a697c4a5203d50573a8cbe392d0ac5dc2163914cf52a3fc967717d3aa83da6bb21c84648e353af0f9ad3f6f912f0ac47a27c8b8c6774429b2c36e17a46e2099cf0a0f443c956570e818608b08702d67855c5f2d3d84cf055f95372b530353eecd9516a799c9fd7f683a5483cb1b30c705d1782162d9ec564a47c3fdecec1caaf546edb6354f557e45a2ad7382b5cfc0aa03f8d8160ea5e0329a35bf4ff8a259fd008a1f0668f29c99c05\")"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#eda\n",
        "dl_data.isnull().sum()#no null values\n",
        "dl_data.shape#no of rows and columns\n",
        "dl_data.describe()#all statistical measures of data\n",
        "dl_data.info()#no encoding required\n",
        "dl_data.drop_duplicates()\n",
        "dl_data.tail(6)\n",
        "dl_data.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 534
        },
        "id": "-U8QX2IDDvp2",
        "outputId": "cde4d4ad-cc1f-42b8-8c36-f267c5163924"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "<class 'pandas.core.frame.DataFrame'>\n",
            "RangeIndex: 9815 entries, 0 to 9814\n",
            "Data columns (total 8 columns):\n",
            " #   Column                  Non-Null Count  Dtype  \n",
            "---  ------                  --------------  -----  \n",
            " 0   Rotational speed [rpm]  9815 non-null   int64  \n",
            " 1   Torque [Nm]             9815 non-null   float64\n",
            " 2   Tool wear [min]         9815 non-null   int64  \n",
            " 3   TWF                     9815 non-null   int64  \n",
            " 4   HDF                     9815 non-null   int64  \n",
            " 5   PWF                     9815 non-null   int64  \n",
            " 6   OSF                     9815 non-null   int64  \n",
            " 7   Machine failure         9815 non-null   int64  \n",
            "dtypes: float64(1), int64(7)\n",
            "memory usage: 613.6 KB\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Rotational speed [rpm]  Torque [Nm]  Tool wear [min]  TWF  HDF  PWF  OSF  \\\n",
              "0                    1551         42.8                0    0    0    0    0   \n",
              "1                    1408         46.3                3    0    0    0    0   \n",
              "2                    1498         49.4                5    0    0    0    0   \n",
              "3                    1433         39.5                7    0    0    0    0   \n",
              "4                    1408         40.0                9    0    0    0    0   \n",
              "\n",
              "   Machine failure  \n",
              "0                0  \n",
              "1                0  \n",
              "2                0  \n",
              "3                0  \n",
              "4                0  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-c76682f9-20f1-49d5-a0b9-e0e6bef8ada8\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Rotational speed [rpm]</th>\n",
              "      <th>Torque [Nm]</th>\n",
              "      <th>Tool wear [min]</th>\n",
              "      <th>TWF</th>\n",
              "      <th>HDF</th>\n",
              "      <th>PWF</th>\n",
              "      <th>OSF</th>\n",
              "      <th>Machine failure</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>1551</td>\n",
              "      <td>42.8</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1408</td>\n",
              "      <td>46.3</td>\n",
              "      <td>3</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>1498</td>\n",
              "      <td>49.4</td>\n",
              "      <td>5</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>1433</td>\n",
              "      <td>39.5</td>\n",
              "      <td>7</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>1408</td>\n",
              "      <td>40.0</td>\n",
              "      <td>9</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "      <td>0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-c76682f9-20f1-49d5-a0b9-e0e6bef8ada8')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-c76682f9-20f1-49d5-a0b9-e0e6bef8ada8 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-c76682f9-20f1-49d5-a0b9-e0e6bef8ada8');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ],
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "dataframe",
              "variable_name": "dl_data",
              "summary": "{\n  \"name\": \"dl_data\",\n  \"rows\": 9815,\n  \"fields\": [\n    {\n      \"column\": \"Rotational speed [rpm]\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 147,\n        \"min\": 1168,\n        \"max\": 2076,\n        \"num_unique_values\": 799,\n        \"samples\": [\n          2032,\n          1980,\n          1461\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Torque [Nm]\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 9.420254344733607,\n        \"min\": 16.7,\n        \"max\": 68.9,\n        \"num_unique_values\": 495,\n        \"samples\": [\n          59.2,\n          49.9,\n          55.6\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Tool wear [min]\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 63,\n        \"min\": 0,\n        \"max\": 253,\n        \"num_unique_values\": 246,\n        \"samples\": [\n          93,\n          14,\n          215\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"TWF\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"HDF\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"PWF\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"OSF\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    },\n    {\n      \"column\": \"Machine failure\",\n      \"properties\": {\n        \"dtype\": \"number\",\n        \"std\": 0,\n        \"min\": 0,\n        \"max\": 1,\n        \"num_unique_values\": 2,\n        \"samples\": [\n          1,\n          0\n        ],\n        \"semantic_type\": \"\",\n        \"description\": \"\"\n      }\n    }\n  ]\n}"
            }
          },
          "metadata": {},
          "execution_count": 7
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "#split\n",
        "x=dl_data.drop(columns=[\"Machine failure\"])#all feauture columns\n",
        "y=dl_data[[\"Machine failure\"]]\n",
        "from sklearn.model_selection import train_test_split\n",
        "x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "ss=StandardScaler()\n",
        "ss_fit_transform_feauture_data=ss.fit_transform(x_train)\n",
        "ss_transform_x_test=ss.transform(x_test)\n",
        "\n",
        "#after scaling\n",
        "sb.histplot(ss_fit_transform_feauture_data[:,0])\n",
        "plt.title(\"after scaling\")\n",
        "plt.show()\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 254
        },
        "id": "97e8lTiWESkv",
        "outputId": "53361a66-8b58-4b5a-8eb9-1707d94ab89b"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAj4AAAGzCAYAAAAv9B03AAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjAsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvlHJYcgAAAAlwSFlzAAAPYQAAD2EBqD+naQAALr9JREFUeJzt3X1UVXW+x/HPAXlQ8UAoD3JVfCqV1JzBp5MzV0UGUnIyWd2p5XWorCYCRmXSxhlTs2lsOSVWQ9p0Hay5skybtHwYn8iHSkilZSMycbPRcESgMkGRJznn/jGLM50RAfFwzoH9fq2112L/9m+f8917JXz67d/e22Sz2WwCAAAwAC93FwAAAOAqBB8AAGAYBB8AAGAYBB8AAGAYBB8AAGAYBB8AAGAYBB8AAGAYBB8AAGAYBB8AAGAYBB8AN+3zzz9XXFycAgMDZTKZtHXrVneX5BKTJk3SpEmT7OtnzpyRyWTS+vXr3VYTgOZ1cXcBADq+pKQknT59Ws8995yCgoI0evRoZWdnq7y8XPPmzXN3eQBgZ+JdXQBuRnV1tbp166Zf//rX+s1vfmNvv/vuu1VQUKAzZ864r7h21jjac+DAAUmSzWZTbW2tfHx85O3t7b7CAFwXl7oA3JSvvvpKkhQUFNTu32W1WlVTU9Pu39NWJpNJ/v7+hB7AgxF8ADTpyy+/1BNPPKEhQ4aoa9eu6tmzp+677z6HEZxly5YpMjJSkrRgwQKZTCb1799fkyZN0o4dO/Tll1/KZDLZ2xvV1tZq6dKlGjx4sPz8/NS3b18tXLhQtbW1DjWYTCalpqZqw4YNuv322+Xn56ddu3Zdt+Zjx44pPj5evXr1UteuXTVgwAA9/PDDDn2sVqteeukljRgxQv7+/goJCdFdd92lY8eO2ftkZWUpJiZGoaGh8vPzU1RUlNasWdPiOWtqjs+DDz6ogIAAnTt3TjNmzFBAQIBCQkL05JNPqqGhwWH/b775RrNnz5bZbFZQUJCSkpL06aefMm8IcCLm+ABo0tGjR3X48GHdf//96tOnj86cOaM1a9Zo0qRJKiwsVLdu3TRz5kwFBQVp/vz5euCBBzRt2jQFBASoe/fuqqio0D/+8Q9lZGRIkgICAiT9M3j8+Mc/1ocffqjHHntMw4YN04kTJ5SRkaH/+7//u2Zi9Pvvv69NmzYpNTVVvXr1cghQ31VeXq64uDiFhITol7/8pYKCgnTmzBm98847Dv3mzJmj9evXa+rUqXrkkUd09epVffDBB8rLy9Po0aMlSWvWrNHtt9+uH//4x+rSpYu2bdumJ554QlarVSkpKTd8LhsaGhQfH69x48bphRde0L59+/Tiiy9q0KBBSk5Otp+X6dOn68iRI0pOTtbQoUP17rvvKikp6Ya/D0AzbADQhCtXrlzTlpuba5Nke/PNN+1tp0+ftkmy/e53v3Pom5CQYIuMjLzmM/70pz/ZvLy8bB988IFD+9q1a22SbB999JG9TZLNy8vLdvLkyRbr3bJli02S7ejRo9ft8/7779sk2X7+859fs81qtdp/burY4+PjbQMHDnRomzhxom3ixIn29cZzkZWVZW9LSkqySbItX77cYd/vfe97tujoaPv6n//8Z5sk2+rVq+1tDQ0NtpiYmGs+E0DbcakLQJO6du1q/7m+vl7ffPONBg8erKCgIH3yySdt/tzNmzdr2LBhGjp0qL7++mv7EhMTI0nav3+/Q/+JEycqKiqqxc9tnGO0fft21dfXN9nnz3/+s0wmk5YuXXrNNpPJZP/5u8deUVGhr7/+WhMnTtTf//53VVRUtFhLUx5//HGH9R/+8If6+9//bl/ftWuXfHx89Oijj9rbvLy82jTCBOD6CD4AmlRdXa0lS5aob9++8vPzU69evRQSEqKLFy+2+Y+/9M9n/pw8eVIhISEOy2233Sbpn5esvmvAgAGt+tyJEycqMTFRzzzzjHr16qV77rlHWVlZDvOGvvjiC0VERCg4OLjZz/roo48UGxur7t27KygoSCEhIfrVr34lSW069sa5RN91yy236Ntvv7Wvf/nll+rdu7e6devm0G/w4ME3/H0Aro85PgCalJaWpqysLM2bN08Wi8X+cML7779fVqu1zZ9rtVo1YsQIrVq1qsntffv2dVj/7uhLc0wmk95++23l5eVp27Zt2r17tx5++GG9+OKLysvLs88xaskXX3yhKVOmaOjQoVq1apX69u0rX19f7dy5UxkZGW06du7yAjwHwQdAk95++20lJSXpxRdftLfV1NTo4sWLrdr/u5eOvmvQoEH69NNPNWXKlOv2uRnjx4/X+PHj9dxzzyk7O1uzZs3Sxo0b9cgjj2jQoEHavXu3Lly4cN1Rn23btqm2tlbvvfee+vXrZ2//90twzhYZGan9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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "\n",
        "\n",
        "#before scaling\n",
        "plt.figure(figsize=(10,6))\n",
        "sb.histplot(x_train.iloc[:,0])\n",
        "plt.title(\"before scaling\")\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 237
        },
        "outputId": "d436d2ab-9d46-4dde-df64-6234d8a27dc4",
        "id": "ewTdhqtvF4gj"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "Text(0.5, 1.0, 'before scaling')"
            ]
          },
          "metadata": {},
          "execution_count": 14
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 1000x600 with 1 Axes>"
            ],
            "image/png": "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ZuiVV16p8fND3fFT19oPrVy5Ujt27FBsbOxF+7jWUBVXuta+/vpr3XbbbUpMTNSWLVv05ZdfaurUqQoKCnK1GT9+vN5//32988472rp1q44dO6aBAwe69l+4cEF9+/ZVaWmptm/frtdee03Lly/XtGnTavz8UHdc6VqbMGGC1qxZozfeeEPZ2dkaN26cRo8erffee8/VhmsNVbF161alp6drx44dWr9+vcrKytS7d2+dO3fO1aY6rqXDhw+rb9++uuuuu7Rnzx6NGzdOv/nNb7R27dpaPd9Gx0C9J8lYuXLlT7b597//bUgyjhw5YhiGYRw4cMCQZOzatcvV5qOPPjIsFovxzTffGIZhGIsXLzaaNWtmlJSUuNpMmjTJaN++ffWfBOqFy11r//3vf41rrrnG2LdvnxEXF2csWLDAtY9rDZ641LX2wAMPGGlpaZftc/r0acPf39945513XNuys7MNSUZmZqZhGIbx4YcfGj4+PkZBQYGrzcsvv2xYrVa36w+Nx6WutY4dOxqzZs1y23bDDTcYzz77rGEYXGvw3IkTJwxJxtatWw3DqL5r6emnnzY6duzoNtYDDzxgpKSk1PQpNWrMSDUShYWFslgsCg8PlyRlZmYqPDxcN954o6tNr1695OPjo507d7ra3HHHHQoICHC1SUlJUW5urr777rtarR91V0VFhYYOHaqJEyeqY8eOF+3nWkN1qKio0AcffKDrrrtOKSkpioqKUo8ePdxuycrKylJZWZl69erl2paYmCibzabMzExJ319rnTt3dnthe0pKipxOp/bv319r54O67ZZbbtF7772nb775RoZhaPPmzfrqq6/Uu3dvSVxr8FxhYaEkKSIiQlL1XUuZmZlux6hsU3kM1AyCVCNw/vx5TZo0SQ899JCsVqskqaCgQFFRUW7t/Pz8FBERoYKCAlebH/5LK8n1vbIN8Mc//lF+fn4aO3bsJfdzraE6nDhxQmfPntXcuXPVp08frVu3Tvfff78GDhyorVu3Svr+WgkICHD9wqhSdHQ01xpMWbRokTp06KBWrVopICBAffr0UUZGhu644w5JXGvwTEVFhcaNG6dbb71VnTp1klR919Ll2jidThUXF9fE6UCSn7cLQM0qKyvToEGDZBiGXn75ZW+XgwYmKytLL774oj7//HNZLBZvl4MGrKKiQpJ03333afz48ZKk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\n"
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import Dense\n",
        "model=Sequential([\n",
        "    Dense(32,activation=\"relu\",input_shape=(x_train.shape[1],)), #inpiut layers\n",
        "    Dense(16,activation=\"relu\",),#hidden layers\n",
        "    Dense(1,activation=\"sigmoid\")#output layers\n",
        "])"
      ],
      "metadata": {
        "id": "78wr6Jn4GKzY",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "6789fedd-bb02-4980-a465-28e2c80930b2"
      },
      "execution_count": 18,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.12/dist-packages/keras/src/layers/core/dense.py:106: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
            "  super().__init__(activity_regularizer=activity_regularizer, **kwargs)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])"
      ],
      "metadata": {
        "id": "Gvu8s-9iGJY3"
      },
      "execution_count": 19,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "model.fit(x_train,y_train,epochs=20,batch_size=32)\n",
        "model.evaluate(x_test,y_test)\n",
        "model.predict(x_test)"
      ],
      "metadata": {
        "id": "45oAyE1YHDRw",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "603eba6d-ba04-41c0-ffdc-ba9007ffe446"
      },
      "execution_count": 21,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/20\n",
            "\u001b[1m246/246\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.9879 - loss: 0.0445\n",
            "Epoch 2/20\n",
            "\u001b[1m246/246\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.9882 - loss: 0.0610\n",
            "Epoch 3/20\n",
            "\u001b[1m246/246\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.9864 - loss: 0.0655\n",
            "Epoch 4/20\n",
            "\u001b[1m246/246\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.9922 - loss: 0.0385\n",
            "Epoch 5/20\n",
            "\u001b[1m246/246\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.9931 - loss: 0.0354\n",
            "Epoch 6/20\n",
            "\u001b[1m246/246\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.9761 - loss: 0.5370\n",
            "Epoch 7/20\n",
            "\u001b[1m246/246\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.9794 - loss: 0.2477\n",
            "Epoch 8/20\n",
            "\u001b[1m246/246\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.9911 - loss: 0.0414\n",
            "Epoch 9/20\n",
            "\u001b[1m246/246\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.9917 - loss: 0.0391\n",
            "Epoch 10/20\n",
            "\u001b[1m246/246\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.9935 - loss: 0.0328\n",
            "Epoch 11/20\n",
            "\u001b[1m246/246\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.9938 - loss: 0.0256\n",
            "Epoch 12/20\n",
            "\u001b[1m246/246\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.9925 - loss: 0.0418\n",
            "Epoch 13/20\n",
            "\u001b[1m246/246\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.9945 - loss: 0.0265\n",
            "Epoch 14/20\n",
            "\u001b[1m246/246\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 3ms/step - accuracy: 0.9954 - loss: 0.0262\n",
            "Epoch 15/20\n",
            "\u001b[1m246/246\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.9969 - loss: 0.0177\n",
            "Epoch 16/20\n",
            "\u001b[1m246/246\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.9864 - loss: 0.1208\n",
            "Epoch 17/20\n",
            "\u001b[1m246/246\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 3ms/step - accuracy: 0.9890 - loss: 0.0498\n",
            "Epoch 18/20\n",
            "\u001b[1m246/246\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 3ms/step - accuracy: 0.9860 - loss: 0.1445\n",
            "Epoch 19/20\n",
            "\u001b[1m246/246\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.9899 - loss: 0.0471\n",
            "Epoch 20/20\n",
            "\u001b[1m246/246\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 2ms/step - accuracy: 0.9978 - loss: 0.0167\n",
            "\u001b[1m62/62\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 3ms/step - accuracy: 0.9990 - loss: 0.0093\n",
            "\u001b[1m62/62\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 2ms/step\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([[3.61681856e-07],\n",
              "       [1.01720825e-05],\n",
              "       [1.23433158e-04],\n",
              "       ...,\n",
              "       [4.68456747e-05],\n",
              "       [1.34914269e-04],\n",
              "       [4.51930418e-06]], dtype=float32)"
            ]
          },
          "metadata": {},
          "execution_count": 21
        }
      ]
    }
  ]
}
