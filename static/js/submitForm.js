const submitForm = async (event) => {
    event.preventDefault();

    const selectedRadio = document.querySelector('input[name=model_type]:checked');

    if (!selectedRadio){
        alert('Please select a vector type');
        return;
    }

    const data = {
        radio: selectedRadio.value,
    };
    const formData = new FormData(event.target);
    formData.forEach((value, key) => {
        data[key] = value;
    });
    try {
        const response = await fetch('/submit', {
            method: 'PUT',
            body: JSON.stringify(data),
            headers: {
                'Content-Type': 'application/json'
            }
        });
        if (response.ok) {
            window.location.href = '/results';
        } else {
            alert('The song you are searching for might not exist...');
        }
    } catch (error) {
        console.error(error);
        alert('There was an error searching for the song, please try again.');
    }
};

const form = document.getElementById("form");
form.addEventListener("submit", submitForm);