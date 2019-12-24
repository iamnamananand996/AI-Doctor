$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();
    $('#result_data').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict_skin',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (preds) {
                // Get and display the result
                $('.loader').hide();
                $('#result').fadeIn(600);
                $('#result_data').show();
                var result = preds.split(':');
                if (result[0] == 1){
                    res = "Melanocytic nevi";
                }
                else if( result[1] == 1){
                    res = "Melanoma"
                } 
                else if (result[2] == 1){
                    res = "Benign keratosis-like lesions";
                }   
                else if (result[3] == 1){
                    res = "Basal cell carcinoma"
                }   
                else if (result[4] == 1){
                    res = "Actinic keratoses";
                }   
                else if (result[5] == 1){
                    res = "Vascular lesions";
                }
                else{
                    res = "Dermatofibroma";
                }
                   
                $('#result').text(' Result:  ' + res);
                console.log('Success!');
                
                $("#Melanocytic_nevi").text(result[0]);
                $("#Melanoma").text(result[1]);
                $("#Benign_keratosis_like_lesions").text(result[2]);
                $("#Basal_cell_carcinoma").text(result[3]);
                $("#Actinic_keratoses").text(result[4]);
                $("#Vascular_lesions").text(result[5]);
                $("#Dermatofibrom").text(result[6]);
            },
        });
    });

});
