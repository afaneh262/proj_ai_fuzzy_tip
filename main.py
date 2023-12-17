"""
@author: Wajed Afaneh
"""
import PySimpleGUI as sg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def trapmf(x, params):
    assert len(params) == 4, 'abcd parameter must have exactly four elements.'
    a, b, c, d = np.r_[params]
    assert a <= b and b <= c and c <= d, 'abcd requires the four elements \
                                          a <= b <= c <= d.'
    y = np.ones(len(x))

    idx = np.nonzero(x <= b)[0]
    y[idx] = trimf(x[idx], np.r_[a, b, b])

    idx = np.nonzero(x >= c)[0]
    y[idx] = trimf(x[idx], np.r_[c, c, d])

    idx = np.nonzero(x < a)[0]
    y[idx] = np.zeros(len(idx))

    idx = np.nonzero(x > d)[0]
    y[idx] = np.zeros(len(idx))

    return y

def gaussmf(x, mean, sigma):
    return np.exp(-((x - mean)**2.) / (2 * sigma**2.))

def trimf(x, params):
    assert len(params) == 3, 'abc parameter must have exactly three elements.'
    a, b, c = np.r_[params]     # Zero-indexing in Python
    assert a <= b and b <= c, 'abc requires the three elements a <= b <= c.'

    y = np.zeros(len(x))

    # Left side
    if a != b:
        idx = np.nonzero(np.logical_and(a < x, x < b))[0]
        y[idx] = (x[idx] - a) / float(b - a)

    # Right side
    if b != c:
        idx = np.nonzero(np.logical_and(b < x, x < c))[0]
        y[idx] = (c - x[idx]) / float(c - b)

    idx = np.nonzero(x == b)
    y[idx] = 1
    return y
def interp_membership(x, mf, x_val):
    return np.interp(x_val, x, mf)

def defuzz(x, mf, method='centroid'):
    if method == 'centroid':
        return np.sum(x * mf) / np.sum(mf)
    # Add more defuzzification methods if needed
    else:
        raise ValueError("Invalid defuzzification method")

def fuzzy_logic_tip(service_quality, food_quality, window):
    plt.switch_backend('TkAgg')

    # Food Quality and service ranges [0, 10]
    # Tip has a range of [0, 30]
    x_food_qual = np.arange(0, 11, 1)
    x_serv = np.arange(0, 11, 1)
    x_tip = np.arange(0, 31, 1)

    # Generate fuzzy membership functions for food quality
    food_qual_rancid = trapmf(x_food_qual, [0, 0, 3, 6])
    food_qual_delicious = trapmf(x_food_qual, [4, 7, 10, 10])

    # Generate fuzzy membership functions for service
    serv_poor = gaussmf(x_serv, 0, 1.7)
    serv_good = gaussmf(x_serv, 5, 1.7)
    serv_excellent = gaussmf(x_serv, 10, 1.7)

    # Generate fuzzy membership functions for tip
    tip_cheap = trimf(x_tip, [0, 5, 10])
    tip_average = trimf(x_tip, [10, 15, 20])
    tip_generous = trimf(x_tip, [20, 25, 30])

    # Visualize these universes and membership functions
    fig, (ax0, ax1, ax2, ax4, ax5) = plt.subplots(nrows=5, figsize=(8, 9))

    ax0.plot(x_food_qual, food_qual_rancid, 'b', linewidth=1, label='Rancid')
    ax0.plot(x_food_qual, food_qual_delicious, 'r', linewidth=1, label='Delicious')
    ax0.set_title('Food quality')
    ax0.legend()

    ax1.plot(x_serv, serv_poor, 'b', linewidth=1, label='Poor')
    ax1.plot(x_serv, serv_good, 'g', linewidth=1, label='Good')
    ax1.plot(x_serv, serv_excellent, 'r', linewidth=1, label='Excellent')
    ax1.set_title('Service quality')
    ax1.legend()

    ax2.plot(x_tip, tip_cheap, 'b', linewidth=1, label='Cheap')
    ax2.plot(x_tip, tip_average, 'g', linewidth=1, label='Average')
    ax2.plot(x_tip, tip_generous, 'r', linewidth=1, label='Generous')
    ax2.set_title('Tip amount')
    ax2.legend()

    # We need the activation of our fuzzy membership functions at provided values
    food_qual_level_rancid = interp_membership(x_food_qual, food_qual_rancid, food_quality)
    food_qual_level_delicious = interp_membership(x_food_qual, food_qual_delicious, food_quality)

    serv_qual_level_poor = interp_membership(x_serv, serv_poor, service_quality)
    serv_qual_level_good = interp_membership(x_serv, serv_good, service_quality)
    serv_qual_level_excellent = interp_membership(x_serv, serv_excellent, service_quality)

    # Apply rules
    # Rule 1: If the service is poor or the food is rancid, then tip is cheap.
    # The OR operator means we take the maximum of these two.
    active_rule1 = np.fmax(food_qual_level_rancid, serv_qual_level_poor)
    tip_activation_cheap = np.fmin(active_rule1, tip_cheap)  # removed entirely to 0

    # Rule 2: If the service is good, then tip is average.
    tip_activation_average = np.fmin(serv_qual_level_good, tip_average)

    # Rule 3: If the service is excellent or the food is delicious, then tip is generous.
    active_rule3 = np.fmax(food_qual_level_delicious, serv_qual_level_excellent)
    tip_activation_generous = np.fmin(active_rule3, tip_generous)

    tip0 = np.zeros_like(x_tip)

    # Visualize this
    ax4.fill_between(x_tip, tip0, tip_activation_cheap, facecolor='b', alpha=0.7)
    ax4.plot(x_tip, tip_cheap, 'b', linewidth=0.5, linestyle='--', )
    ax4.fill_between(x_tip, tip0, tip_activation_average, facecolor='g', alpha=0.7)
    ax4.plot(x_tip, tip_average, 'g', linewidth=0.5, linestyle='--')
    ax4.fill_between(x_tip, tip0, tip_activation_generous, facecolor='r', alpha=0.7)
    ax4.plot(x_tip, tip_generous, 'r', linewidth=0.5, linestyle='--')
    ax4.set_title('Output membership activity')

    # Aggregate all three output membership functions together
    aggregated = np.fmax(np.fmax(tip_activation_cheap, tip_activation_average), tip_activation_generous)

    # Calculate defuzzified result
    tip = defuzz(x_tip, aggregated, 'centroid')
    tip_activation = interp_membership(x_tip, aggregated, tip)

    # Visualize
    ax5.plot(x_tip, tip_cheap, 'b', linewidth=0.5, linestyle='--', )
    ax5.plot(x_tip, tip_average, 'g', linewidth=0.5, linestyle='--')
    ax5.plot(x_tip, tip_generous, 'r', linewidth=0.5, linestyle='--')
    ax5.fill_between(x_tip, tip0, aggregated, facecolor='Black', alpha=0.7)
    ax5.plot([tip, tip], [0, tip_activation], 'k', linewidth=1, alpha=0.9)
    ax5.set_title('Aggregated membership and result (line)')

    plt.tight_layout()

    # Embed the Matplotlib figure in the PySimpleGUI window
    canvas_elem = window['canvas']
    canvas = canvas_elem.Widget

    # Clear the previous drawing
    for widget in canvas.winfo_children():
        widget.destroy()

    # Draw the canvas
    img = FigureCanvasTkAgg(fig, master=canvas)
    img.draw()
    img.get_tk_widget().pack(side='top', fill='both', expand=1)

    window['output'].update(f'Tip Percentage: {tip:.2f}')


def create_layout():
    layout = [
        [sg.Text('Food Quality:'), sg.Slider(range=(0, 10), orientation='h', size=(20, 10), default_value=5, key='food')],
        [sg.Text('Service Quality:'), sg.Slider(range=(0, 10), orientation='h', size=(20, 10), default_value=5, key='service')],
        [sg.Button('Calculate Tip'), sg.Button('Exit')],
        [sg.Text('', size=(20, 1), key='output')],
        [sg.Canvas(key='canvas')],
    ]
    return layout

def main():
    sg.theme('Default1')
    window = sg.Window('Tip Fuzzy', create_layout(), resizable=True)

    while True:
        event, values = window.read()

        if event == sg.WINDOW_CLOSED or event == 'Exit':
            break

        if event == 'Calculate Tip':
            service_quality = (values['service'] / 10.0) * 10 
            food_quality = (values['food'] / 10.0) * 10
            fuzzy_logic_tip(service_quality, food_quality, window)

    window.close()

if __name__ == '__main__':
    main()
