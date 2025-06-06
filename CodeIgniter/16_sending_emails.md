# **How to Send Emails Using CodeIgniter’s Email Library?**  

CodeIgniter provides a built-in **Email Library** that allows you to send emails via SMTP, Sendmail, or PHP's mail function. This guide will walk you through configuring and sending emails in CodeIgniter.  

---

## **1. Configure Email Settings**  

📁 Open `app/Config/Email.php` and set up your email configuration:  

```php
<?php

namespace Config;

use CodeIgniter\Config\BaseConfig;

class Email extends BaseConfig
{
    public $fromEmail = 'your_email@gmail.com';
    public $fromName  = 'Your Name';
    public $SMTPHost  = 'smtp.gmail.com';
    public $SMTPUser  = 'your_email@gmail.com';
    public $SMTPPass  = 'your_app_password'; // Use App Password if 2FA is enabled
    public $SMTPPort  = 587;
    public $SMTPCrypto = 'tls'; // Use 'ssl' for port 465
    public $protocol  = 'smtp';
    public $mailType  = 'html'; // Use 'text' for plain emails
}
```
✅ This configures SMTP using **Gmail**. You can replace these values for other email providers like **Outlook, Yahoo, etc.**  

---

## **2. Create an Email Controller**  

📁 `app/Controllers/EmailController.php`  

```php
<?php
namespace App\Controllers;

use CodeIgniter\Controller;
use CodeIgniter\Email\Email;

class EmailController extends Controller
{
    public function send()
    {
        $email = service('email');

        $email->setTo('recipient@example.com');
        $email->setFrom('your_email@gmail.com', 'Your Name');
        $email->setSubject('Test Email from CodeIgniter');
        $email->setMessage('<h2>Hello from CodeIgniter!</h2><p>This is a test email.</p>');

        if ($email->send()) {
            return 'Email sent successfully!';
        } else {
            return 'Failed to send email: ' . $email->printDebugger();
        }
    }
}
```
✅ This **loads the Email library**, sets the recipient, subject, and body, and then **sends the email**.  

---

## **3. Add Routes for Sending Email**  

📁 `app/Config/Routes.php`  

```php
$routes->get('/send-email', 'EmailController::send');
```
✅ Now, visiting `http://localhost:8080/send-email` will trigger email sending.  

---

## **4. Testing the Email Functionality**  

1. **Start the local server:**  
   ```sh
   php spark serve
   ```
2. Open your browser and visit:  
   ```
   http://localhost:8080/send-email
   ```
3. If successful, you will see:  
   ```
   Email sent successfully!
   ```
4. Check your **recipient’s inbox** (or spam folder).

---

## **5. Handling Attachments**  

To send **attachments**, modify the `send()` function:  

```php
$email->attach(WRITEPATH . 'uploads/sample.pdf');
```
✅ This attaches a file from `writable/uploads/sample.pdf`.  

---

## **6. Sending Emails to Multiple Recipients**  

```php
$email->setTo(['recipient1@example.com', 'recipient2@example.com']);
$email->setCC('cc@example.com');
$email->setBCC('bcc@example.com');
```
✅ Sends an email to multiple recipients with CC and BCC.  

---

## **7. Troubleshooting Errors**  

🔹 **Email Not Sending?**  
✔ Ensure SMTP settings are correct.  
✔ Check for **blocked less secure apps** (Google users may need an **App Password**).  
✔ Use `printDebugger()` to debug:  

```php
echo $email->printDebugger(['headers']);
```

---

## **Next Steps**  

🚀 **Want to use a different mail provider (SendGrid, Mailgun)?** I can help! 😊