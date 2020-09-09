jsPsych.plugins['survey-multi-type'] = (function() {

  var plugin = {};

  plugin.trial = function(display_element, trial) {

    trial.preamble = typeof trial.preamble == 'undefined' ? "" : trial.preamble;

    // Parameters for the text box
    if (typeof trial.rows == 'undefined') {
      trial.rows = [];
      for (var i = 0; i < trial.questions.length; i++) {
        trial.rows.push(3);
      }
    }
    
    if (typeof trial.columns == 'undefined') {
      trial.columns = [];
      for (var i = 0; i < trial.questions.length; i++) {
        trial.columns.push(60);
      }
    }

    display_element.html('');

    // Parameters for the buttons
    trial.button_html = trial.button_html || '<button class="jspsych-btn">%choice%</button>';
    trial.response_ends_trial = (typeof trial.response_ends_trial === 'undefined') ? true : trial.response_ends_trial;
    trial.timing_stim = trial.timing_stim || -1; // if -1, then show indefinitely
    trial.timing_response = trial.timing_response || -1; // if -1, then wait for response forever
    trial.is_html = (typeof trial.is_html === 'undefined') ? false : trial.is_html;
    trial.prompt = (typeof trial.prompt === 'undefined') ? "" : trial.prompt;
    var plugin_id_name = "jspsych-survey-multi-choice";
    var plugin_id_selector = '#' + plugin_id_name;
    var _join = function( /*args*/ ) {
      var arr = Array.prototype.slice.call(arguments, _join.length);
      return arr.join(separator = '-');
    }
    trial.required = typeof trial.required == 'undefined' ? null : trial.required;
    trial.horizontal = typeof trial.required == 'undefined' ? false : trial.horizontal;

    //For all of them together
    trial = jsPsych.pluginAPI.evaluateFunctionParameters(trial);

    // show preamble text
    display_element.append($('<div>', {
      "id": 'jspsych-survey-text-preamble',
      "class": 'jspsych-survey-text-preamble'
    }));

    $('#jspsych-survey-text-preamble').html(trial.preamble);

    // display stimulus which in our case is the text talking about the quiz
    if (!trial.is_html) {
      display_element.append($('<img>', {
        src: trial.stimulus,
        id: 'jspsych-button-response-stimulus',
        class: 'block-center'
      }));
    } else {
      display_element.append($('<div>', {
        html: trial.stimulus,
        id: 'jspsych-button-response-stimulus',
        class: 'block-center'
      }));
    }

    // Form element for multiple choice questions
    var trial_form_id = _join(plugin_id_name, "form");
    display_element.append($('<form>', {
      "id": trial_form_id
    }));
    var $trial_form = $("#" + trial_form_id);

    // Form preamble
    var preamble_id_name = _join(plugin_id_name, 'preamble');
    $trial_form.append($('<div>', {
      "id": preamble_id_name,
      "class": preamble_id_name
    }));
    $('#' + preamble_id_name).html(trial.preamble);

    var setTimeoutHandlers = [];

    // Multiple-choice questions
    for (var i = 0; i < trial.questions.length; i++) {
      // create question container
      var question_classes = [_join(plugin_id_name, 'question')];
      if (trial.horizontal) {
        question_classes.push(_join(plugin_id_name, 'horizontal'));
      }
      $trial_form.append($('<div>', {
        "id": _join(plugin_id_name, i),
        "class": question_classes.join(' ')
      }));
      var question_selector = _join(plugin_id_selector, i);

      // add question text
      $(question_selector).append(
        '<p class="' + plugin_id_name + '-text survey-multi-choice">' + trial.questions[i] + '</p>'
      );

      // create option radio buttons
      for (var j = 0; j < trial.options[i].length; j++) {
        var option_id_name = _join(plugin_id_name, "option", i, j),
          option_id_selector = '#' + option_id_name;

        // add radio button container
        $(question_selector).append($('<div>', {
          "id": option_id_name,
          "class": _join(plugin_id_name, 'option')
        }));

        // add label and question text
        var option_label = '<label class="' + plugin_id_name + '-text">' + trial.options[i][j] + '</label>';
        $(option_id_selector).append(option_label);

        // create radio button
        var input_id_name = _join(plugin_id_name, 'response', i);
        $(option_id_selector + " label").prepend('<input type="radio" name="' + input_id_name + '" value="' + trial.options[i][j] + '">');
      }

      if (trial.required && trial.required[i]) {
        // add "question required" asterisk
        $(question_selector + " p").append("<span class='required'>*</span>")

        // add required property
        $(question_selector + " input:radio").prop("required", true);
      }
    }

    // Text questions
    for (var i = 0; i < trial.text_questions.length; i++) {
      // create div
      display_element.append($('<div>', {
        "id": 'jspsych-survey-text-' + i,
        "class": 'jspsych-survey-text-question'
      }));

      // add question text
      $("#jspsych-survey-text-" + i).append('<p class="jspsych-survey-text">' + trial.text_questions[i] + '</p>');

      // add text box
      $("#jspsych-survey-text-" + i).append('<textarea name="#jspsych-survey-text-response-' + i + '" cols="' + trial.columns[i] + '" rows="' + trial.rows[i] + '"></textarea>');
    }

    //show prompt if there is one
    if (trial.prompt !== "") {
      display_element.append(trial.prompt);
    }


    //Displaying buttons
    var buttons = [];
    if (Array.isArray(trial.button_html)) {
      if (trial.button_html.length == trial.choices.length) {
        buttons = trial.button_html;
      } else {
        console.error('Error in button-response plugin. The length of the button_html array does not equal the length of the choices array');
      }
    } else {
      for (var i = 0; i < trial.choices.length; i++) {
        buttons.push(trial.button_html);
      }
    }
    display_element.append('<div id="jspsych-button-response-btngroup" class="center-content block-center"></div>')
    for (var i = 0; i < trial.choices.length; i++) {
      var str = buttons[i].replace(/%choice%/g, trial.choices[i]);
      $('#jspsych-button-response-btngroup').append(
        $(str).attr('id', 'jspsych-button-response-button-' + i).data('choice', i).addClass('jspsych-button-response-button').on('click', function(e) {
          var choice = $('#' + this.id).data('choice');
          after_response(choice);
        })
      );
    }
    
    // start time
    var startTime = Date.now();

    // create object to hold choice responses
    var choice_response_data = [];
    $("div." + plugin_id_name + "-question").each(function(index) {
      // var id = "Q" + index;
      var val = $(this).find("input:radio:checked").val();
      // var obje = {};
      // obje[id] = val;
      // $.extend(question_data, obje);
      choice_response_data.push(val)
    });

    var correct = null;
    var mistake = false;
    console.log(trial.correct)
    if (trial.correct) {
      correct = [];
      for (var i = 0; i < choice_response_data.length; i++) {
        var c = trial.correct[i] == choice_response_data[i];
        if (!c) mistake = true;
        correct.push(c)
      }
    }

    // store response
    var button_response = {
      button: -1
    };

    function after_response(choice) {
      button_response.button = choice;
      $("#jspsych-button-response-stimulus").addClass('responded');
      $('.jspsych-button-response-button').off('click').attr('disabled', 'disabled');
      if (trial.response_ends_trial) {
        end_trial();
      }
    };

    // function to end trial when it is time
    function end_trial() {
      for (var i = 0; i < setTimeoutHandlers.length; i++) {
        clearTimeout(setTimeoutHandlers[i]);
      }
      var end_time = Date.now();
      var rt = end_time - startTime;
      button_response.rt = rt;

      var question_data = {};
      $("div.jspsych-survey-text-question").each(function(index) {
        var id = "Q" + index;
        var val = $(this).children('textarea').val();
        var obje = {};
        obje[id] = val;
        $.extend(question_data, obje);
      });

      var trial_data = {
        "rt": button_response.rt,
        "stimulus": trial.stimulus,
        "button_pressed": button_response.button,
        "responses": JSON.stringify(question_data),
        "questions": JSON.stringify(trial.questions),
        "choice_responses": JSON.stringify(choice_response_data),
        "correct": correct
      };

      if (trial.on_mistake && mistake) trial.on_mistake(trial_data);
      display_element.html('');
      jsPsych.finishTrial(trial_data);
    };

    // hide image if timing is set
    if (trial.timing_stim > 0) {
      var t1 = setTimeout(function() {
        $('#jspsych-button-response-stimulus').css('visibility', 'hidden');
      }, trial.timing_stim);
      setTimeoutHandlers.push(t1);
    }

    // end trial if time limit is set
    if (trial.timing_response > 0) {
      var t2 = setTimeout(function() {
        end_trial();
      }, trial.timing_response);
      setTimeoutHandlers.push(t2);
    }

  };

  return plugin;
})();
