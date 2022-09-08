const btn = document.querySelector(".send__data__btn");

btn.addEventListener("click", function () {
	fetch("/coba", {
		headers: {
			"Content-Type": "application/json",
		},
		method: "POST",
		body: JSON.stringify({
			name: "Rahul Kumar",
			country: "India",
		}),
	})
		.then(function (response) {
			if (response.ok) {
				response.json().then(function (response) {
					console.log(response);
				});
			} else {
				throw Error("Something went wrong");
			}
		})
		.catch(function (error) {
			console.log(error);
		});
});
